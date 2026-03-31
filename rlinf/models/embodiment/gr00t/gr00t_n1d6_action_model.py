# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GR00T N1.6 (Gr00tN1d6) model for reinforcement learning action prediction.
Based on gr00t_action_model.py (N1.5) but adapted for the N1.6 architecture:
  - Uses Gr00tN1d6ActionHead (AlternateVLDiT, no future_tokens)
  - Uses Gr00tN1d6Processor + collator for data processing
  - sample_mean_var_val based on N1.6's get_action_with_features
"""

import json
import random
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import torch
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.types import MessageType, VLAStepData
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6, Gr00tN1d6ActionHead
from torch.distributions import Normal
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.gr00t.simulation_io import (
    ACTION_CONVERSION,
    OBS_CONVERSION,
)
from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead


def _rec_to_dtype(x, dtype):
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    return x


class Gr00tN1d6ActionHeadForRL(Gr00tN1d6ActionHead):
    """
    Extends Gr00tN1d6ActionHead with RL-specific capabilities:
    - SDE/CPS/ReinFlow noise for exploration
    - Value head for actor-critic
    - Log-probability computation for policy gradient
    """

    def __init__(
        self,
        config: Gr00tN1d6Config,
        rl_head_config: dict[str, Any],
        output_action_chunks: int,
        valid_action_dim: int,
    ):
        super().__init__(config)
        self.action_chunk = output_action_chunks
        self.rl_config = rl_head_config
        self.padding_value = rl_head_config.padding_value
        self.valid_action_dim = valid_action_dim

        if self.rl_config.use_vlm_value:
            proj_width = 2048
        else:
            proj_width = 3584

        if self.rl_config.add_value_head:
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=(1024, 512, 256),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )

        if self.rl_config.noise_method == "reinflow":
            self.reinflow_explore_noise_net = ExploreNoiseNet(
                in_dim=self.hidden_size,
                out_dim=self.config.max_action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=[0.08, 0.16],
                noise_scheduler_type="learn",
            )

    def get_logprob_norm(self, sample, mu, sigma):
        if self.rl_config.safe_get_logprob:
            dist = Normal(loc=mu, scale=sigma)
            return dist.log_prob(sample)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
            return log_prob

    def sample_mean_var_val(
        self,
        vl_embs: torch.Tensor,
        denoise_steps: int,
        x_t: torch.Tensor,
        embodiment_id: int,
        state_features: torch.Tensor,
        backbone_output: BatchFeature,
        idx: Optional[int | torch.Tensor] = None,
        mode: Literal["train", "eval"] = "train",
        compute_values=False,
    ):
        """
        Sample the mean, variance and value of the action at a given timestep.
        Based on N1.6's get_action_with_features denoising step pattern.

        Key N1.6 differences from N1.5:
        - No future_tokens concatenation
        - Uses AlternateVLDiT with image_mask and backbone_attention_mask
        - Slices model output to [-action_horizon:]
        """
        bsize = vl_embs.shape[0]
        device = vl_embs.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)

        # Noise level for exploration
        if self.rl_config.noise_anneal:
            noise_start, noise_end, anneal_steps = self.rl_config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            noise_level = torch.tensor(self.rl_config.noise_level).to(device)

        # Velocity prediction — N1.6 pattern (no future_tokens)
        t_cont = idx / float(denoise_steps)
        timesteps_tensor = (
            (t_cont * self.num_timestep_buckets).to(torch.int64).to(device)
        )
        action_features = self.action_encoder(x_t, timesteps_tensor, embodiment_id)

        # Position embedding (N1.6 pattern)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=device
            )
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join state and action embeddings (NO future_tokens in N1.6)
        sa_embs = torch.cat((state_features, action_features), dim=1)

        # Run DiT — N1.6 uses AlternateVLDiT with masks
        if self.config.use_alternate_vl_dit:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )

        # Slice to action portion and decode (N1.6 pattern)
        model_output = model_output[:, -self.config.action_horizon :]
        v_t = self.action_decoder(model_output, embodiment_id)

        # ODE/SDE sampling (same as N1.5 — these are RLinf RL additions)
        timesteps = torch.linspace(
            0, 1, denoise_steps + 1, device=device, dtype=vl_embs.dtype
        )
        t_input = timesteps[idx]
        delta = timesteps[idx + 1] - timesteps[idx]
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        # In GR00T: x0 = noise, x1 = data
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if mode == "eval":
            x0_weight = 1 - (t_input + delta)
            x1_weight = t_input + delta
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.rl_config.noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        (1 - timesteps)
                        / torch.where(timesteps == 0, timesteps[1], timesteps)
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = (
                    torch.ones_like(t_input)
                    - (t_input + delta)
                    - sigma_i**2 * delta / (2 * (1 - t_input))
                )
                x1_weight = t_input + delta
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.rl_config.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = (torch.ones_like(t_input) - (t_input + delta)) * cos_term
                x1_weight = t_input + delta
                x_t_std = (1 - (t_input + delta)) * sin_term
            elif self.rl_config.noise_method == "reinflow":
                x0_weight = 1 - (t_input + delta)
                x1_weight = t_input + delta
                x_t_std = self.reinflow_explore_noise_net(model_output)
            else:
                raise ValueError(f"Invalid noise method: {self.rl_config.noise_method}")

        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std

    def get_rl_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
    ) -> BatchFeature:
        """Generate RL actions with log-probs, values, and denoising chains."""
        # Encode features (N1.6 pattern)
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Initial noise
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        x_t = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.max_action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        chains = [x_t]
        log_probs = []

        if self.rl_config.joint_logprob:
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(x_t), torch.ones_like(x_t)
            )
            log_probs.append(initial_log_prob)

        num_steps = self.num_inference_timesteps
        # Determine denoise step for logprob calculation
        if mode == "train":
            if self.rl_config.joint_logprob:
                denoise_inds = torch.arange(num_steps)
            else:
                if self.rl_config.noise_method == "flow_sde":
                    if self.rl_config.ignore_last:
                        denoise_inds = torch.tensor(
                            [random.randint(0, num_steps - 2)] * num_steps
                        )
                    else:
                        denoise_inds = torch.tensor(
                            [random.randint(0, num_steps - 1)] * num_steps
                        )
                elif self.rl_config.noise_method == "flow_cps":
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
                elif self.rl_config.noise_method == "reinflow":
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(batch_size, 1)

        # Denoising loop
        for idx in range(num_steps):
            if idx == denoise_inds[0][idx]:
                x_t_mean, x_t_std = self.sample_mean_var_val(
                    vl_embs=vl_embs,
                    idx=idx,
                    x_t=x_t,
                    embodiment_id=embodiment_id,
                    state_features=state_features,
                    backbone_output=backbone_output,
                    mode="train",
                    denoise_steps=num_steps,
                    compute_values=compute_values,
                )
            else:
                x_t_mean, x_t_std = self.sample_mean_var_val(
                    vl_embs=vl_embs,
                    idx=idx,
                    x_t=x_t,
                    embodiment_id=embodiment_id,
                    state_features=state_features,
                    backbone_output=backbone_output,
                    mode="eval",
                    denoise_steps=num_steps,
                    compute_values=compute_values,
                )

            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)

            chains.append(x_t)
            log_probs.append(log_prob)

        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.action_chunk, : self.valid_action_dim
        ]
        if compute_values:
            values = self.get_value(vl_embs, state_features)
            values = values[:, None]
        else:
            values = torch.zeros((batch_size, 1), device=device, dtype=vl_embs.dtype)

        return BatchFeature(
            data={"action_pred": x_0}
        ), {
            "actions": x_0,
            "action_pred": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

    def forward(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        chains,
        denoise_inds,
        compute_values=True,
    ):
        """Actor forward for PPO: recompute log-probs and values from saved chains."""
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        state_features = self.state_encoder(action_input.state, embodiment_id)
        batch_size = vl_embs.shape[0]

        chains_log_probs = []

        if self.rl_config.joint_logprob:
            num_steps = self.num_inference_timesteps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            chains_log_probs.append(initial_log_prob)
        else:
            num_steps = 1
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(batch_size), denoise_ind]
            chains_next = chains[torch.arange(batch_size), denoise_ind + 1]
            x_t_mean, x_t_std = self.sample_mean_var_val(
                vl_embs=vl_embs,
                idx=denoise_ind,
                x_t=chains_pre,
                embodiment_id=embodiment_id,
                state_features=state_features,
                backbone_output=backbone_output,
                mode="train",
                denoise_steps=self.num_inference_timesteps,
                compute_values=compute_values,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            chains_log_probs.append(log_probs)

        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        if compute_values:
            chains_values = self.get_value(vl_embs, state_features)
            chains_values = chains_values[:, None]
        else:
            chains_values = torch.zeros(
                (batch_size, 1), device=chains_log_probs.device, dtype=vl_embs.dtype
            )
        return chains_log_probs, chains_values

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.bfloat16,
            device=device,
        )

    def get_value(self, vl_embs, state_features):
        bsize = vl_embs.shape[0]
        mask_length = vl_embs.shape[1]
        if self.rl_config.value_vlm_mode == "mean_token":
            prefix_mask = [True] * mask_length
        elif self.rl_config.value_vlm_mode == "last_token":
            prefix_mask = [False] * (mask_length - 1) + [True] * 1
        elif self.rl_config.value_vlm_mode == "first_token":
            prefix_mask = [True] * 1 + [False] * (mask_length - 1)
        vl_embs_value = vl_embs[:, prefix_mask, :]
        vl_embs_value = vl_embs_value.mean(dim=1, keepdim=False)
        state_features_value = state_features.reshape(bsize, -1)
        if self.rl_config.use_vlm_value:
            value_embs = vl_embs_value
        else:
            value_embs = torch.cat((vl_embs_value, state_features_value), dim=1)
        values_vlm = self.value_head(value_embs)[:, 0]
        return values_vlm


class GR00T_N1_6_ForRLActionPrediction(Gr00tN1d6, BasePolicy):
    """
    GR00T N1.6 model for reinforcement learning action prediction.
    Wraps Gr00tN1d6 with RL-specific action head and inference pipeline.
    """

    # FSDP wrap policy uses these to find transformer layer classes to auto-wrap.
    # These must match actual classes in the model — get_module_class_from_name
    # searches all submodules by class name. Only list classes that exist.
    _no_split_modules = [
        "Qwen3DecoderLayer",
        "Siglip2EncoderLayer",
    ]

    def __init__(
        self,
        config: Gr00tN1d6Config,
        rl_head_config: dict[str, Any] = None,
        embodiment_tag: Union[str, EmbodimentTag] = "behavior_r1_pro",
        processor: BaseProcessor = None,
        compute_dtype: torch.dtype = torch.bfloat16,
        denoising_steps: Optional[int] = None,
        obs_converter_type: str = "behavior",
        output_action_chunks: int = 32,
        **kwargs,
    ):
        # Gr00tN1d6.__init__ takes (config, transformers_loading_kwargs)
        super().__init__(config)

        if rl_head_config is None:
            # During from_pretrained, init is called with just config first,
            # then attributes are set. Store defaults to avoid errors.
            return

        self.padding_value = rl_head_config.padding_value
        self._processor = processor
        self.compute_dtype = compute_dtype
        self.output_action_chunks = output_action_chunks

        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        if denoising_steps is not None:
            if hasattr(self, "action_head") and hasattr(
                self.action_head, "num_inference_timesteps"
            ):
                self.action_head.num_inference_timesteps = denoising_steps

        self.obs_convert_fn = OBS_CONVERSION[obs_converter_type]
        self.action_convert_fn = ACTION_CONVERSION[obs_converter_type]

        # Load modality configs and normalization stats
        self.modality_configs = self._processor.get_modality_configs()[
            self.embodiment_tag.value
        ]
        self.collate_fn = self._processor.collator
        language_keys = self.modality_configs["language"].modality_keys
        assert len(language_keys) == 1
        self.language_key = language_keys[0]

        # Calculate valid action dim from processor stats
        self._compute_valid_action_dim()

        # Replace action head with RL version (after super().__init__ creates the base one)
        self.action_head = Gr00tN1d6ActionHeadForRL(
            config, rl_head_config, output_action_chunks, self.valid_action_dim
        )

    def _compute_valid_action_dim(self):
        """Compute the real action dimension from processor action stats."""
        action_keys = self.modality_configs["action"].modality_keys
        valid_action_dim = 0
        for key in action_keys:
            dim = self._processor.state_action_processor.norm_params[
                self.embodiment_tag.value
            ]["action"][key]["dim"].item()
            valid_action_dim += dim
        self.valid_action_dim = valid_action_dim

    @property
    def image_nums(self):
        return len(self.modality_configs["video"].modality_keys)

    def eval(self):
        self._processor.eval()
        super().eval()

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = True,
        use_cache: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Actor forward for PPO training: recompute log-probs and values."""
        # Reconstruct the flat input dict that prepare_input expects.
        # pixel_values/image_sizes were reshaped to [bsize, N_imgs, ...] for splitting;
        # flatten back to [N_imgs*bsize, ...] for the backbone.
        inputs = {
            "state": forward_inputs["state"],
            "embodiment_id": forward_inputs["embodiment_id"],
            "input_ids": forward_inputs["input_ids"],
            "attention_mask": forward_inputs["attention_mask"],
            "pixel_values": forward_inputs["pixel_values"].reshape(
                -1, *forward_inputs["pixel_values"].shape[2:]
            ),
            "image_sizes": forward_inputs["image_sizes"].reshape(
                -1, *forward_inputs["image_sizes"].shape[2:]
            ),
        }
        with torch.autocast(device_type="cuda", dtype=self.compute_dtype):
            backbone_inputs, action_inputs = self.prepare_input(inputs)
            backbone_outputs = self.backbone(backbone_inputs)

        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]
        log_probs, value_t = self.action_head(
            backbone_output=backbone_outputs,
            action_input=action_inputs,
            chains=chains,
            denoise_inds=denoise_inds,
            compute_values=compute_values,
        )

        log_probs = log_probs[
            :,
            :,
            : self.action_head.action_chunk,
            : self.valid_action_dim,
        ]
        # Post process
        if self.action_head.rl_config.joint_logprob:
            log_probs = log_probs.mean(dim=1)
            prev_logprobs = kwargs["prev_logprobs"].mean(dim=1)
        else:
            bsize = log_probs.shape[0]
            log_probs = log_probs[:, 0]
            prev_logprobs = kwargs["prev_logprobs"]
            prev_logprobs = prev_logprobs[
                torch.arange(bsize),
                denoise_inds[:, 0],
                : self.action_head.action_chunk,
                : self.valid_action_dim,
            ]
        value_t = value_t.mean(dim=-1, keepdim=False)

        return {
            "logprobs": log_probs.float(),
            "prev_logprobs": prev_logprobs.float(),
            "values": value_t,
            "entropy": torch.zeros_like(log_probs.float()),
        }

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Convert env observations to GR00T format, run RL inference, return actions.

        Uses N1.6 processor pipeline:
        1. Convert env obs → GR00T named keys
        2. Process through Gr00tN1d6Processor → normalized inputs
        3. Collate → model input batch
        4. Backbone → RL action head
        5. Decode actions back to env format
        """
        # Convert env obs to GR00T format
        env_obs["states"] = env_obs["states"].to(torch.bfloat16)
        env_obs["states"] = env_obs["states"].cpu().float()

        # Debug: log state shape on first call
        if not hasattr(self, "_logged_state_shape"):
            print(f"[GR00T N1.6] env_obs states shape: {env_obs['states'].shape}")
            self._logged_state_shape = True

        groot_obs = self.obs_convert_fn(env_obs)

        # Build per-sample observations for the N1.6 processor
        batch_size = groot_obs[
            "video.observation.images.rgb.head_256_256"
        ].shape[0]

        processed_inputs = []
        per_sample_states = []
        for i in range(batch_size):
            # Build VLAStepData for this sample
            images = {}
            states = {}
            for key in self.modality_configs["video"].modality_keys:
                # groot_obs video: [B, T, H, W, C] — extract sample i
                images[key] = groot_obs[f"video.{key}"][i]  # [T, H, W, C]
            for key in self.modality_configs["state"].modality_keys:
                states[key] = groot_obs[f"state.{key}"][i]  # [T, D]
            per_sample_states.append(states)

            language = groot_obs["annotation.human.coarse_action"][i]
            vla_step = VLAStepData(
                images=images,
                states=states,
                actions={},
                text=language,
                embodiment=self.embodiment_tag,
            )
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step}]
            processed_inputs.append(self._processor(messages))

        # Collate and move to model
        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        # Run through model
        normalized_action, result = self._get_rl_action(collated_inputs, mode=mode)

        # Decode actions
        unnormalized_action = self._decode_actions(
            normalized_action, per_sample_states
        )

        # Convert to env action format
        raw_action = self.action_convert_fn(
            unnormalized_action, chunk_size=self.output_action_chunks
        )

        return torch.from_numpy(raw_action), result

    def _get_rl_action(
        self,
        collated_inputs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ):
        """Run backbone then RL action head, return normalized actions + RL metadata."""
        # Collator wraps data under 'inputs' key — unwrap for prepare_input
        inputs = collated_inputs.get("inputs", collated_inputs)
        with torch.autocast(device_type="cuda", dtype=self.compute_dtype):
            backbone_inputs, action_inputs = self.prepare_input(inputs)
            backbone_outputs = self.backbone(backbone_inputs)

            action_head_outputs, rlinf_outputs = self.action_head.get_rl_action(
                backbone_outputs, action_inputs, mode=mode
            )
        actions = rlinf_outputs["actions"].float()

        # Build forward_inputs for later PPO training pass.
        # The rollout worker calls torch.split(value, sizes, dim=0) on every value,
        # so all tensors must have dim 0 = batch_size.
        # Follow N1.5 pattern: reshape pixel_values/image_sizes to [bsize, N_imgs, ...]
        bsize = actions.shape[0]
        forward_inputs = {
            "chains": rlinf_outputs["chains"],
            "denoise_inds": rlinf_outputs["denoise_inds"],
            "state": inputs["state"],
            "embodiment_id": inputs["embodiment_id"].unsqueeze(0).expand(bsize)
                if inputs["embodiment_id"].dim() == 0
                else inputs["embodiment_id"],
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        # pixel_values: list of [C, H, W] tensors → [bsize, N_imgs, C, H, W]
        if isinstance(inputs["pixel_values"], list):
            pv = torch.stack(inputs["pixel_values"])  # [N_imgs, C, H, W]
            forward_inputs["pixel_values"] = pv.unsqueeze(0).expand(
                bsize, *pv.shape
            )
        else:
            forward_inputs["pixel_values"] = inputs["pixel_values"].reshape(
                bsize, self.image_nums, *inputs["pixel_values"].shape[1:]
            )
        # image_sizes: [N_imgs, 2] → [bsize, N_imgs, 2]
        forward_inputs["image_sizes"] = inputs["image_sizes"].unsqueeze(0).expand(
            bsize, *inputs["image_sizes"].shape
        )

        result = {
            "prev_logprobs": rlinf_outputs["prev_logprobs"],
            "prev_values": rlinf_outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }

        return actions, result

    def _decode_actions(
        self, normalized_action: torch.Tensor, per_sample_states: list[dict]
    ) -> dict[str, np.ndarray]:
        """Decode normalized action tensor back to named action dict."""
        action_np = normalized_action.cpu().numpy()
        batch_size = action_np.shape[0]

        # Stack states for batch decoding
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack(
                [s[k] for s in per_sample_states], axis=0
            )

        # Decode each sample
        all_decoded = []
        for i in range(batch_size):
            sample_state = {k: v[i] for k, v in batched_states.items()}
            decoded = self._processor.decode_action(
                action_np[i], self.embodiment_tag, state=sample_state
            )
            all_decoded.append(decoded)

        # Stack into batched dict
        result = {}
        for key in all_decoded[0]:
            result[f"action.{key}"] = np.stack(
                [d[key] for d in all_decoded], axis=0
            )
        return result
