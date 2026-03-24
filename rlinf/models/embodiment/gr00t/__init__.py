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

import torch
from omegaconf import DictConfig


def _apply_embodiment_patches():
    """Patch gr00t embodiment tags to include RLinf-added tags."""
    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EmbodimentTag",
        "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
    )
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING",
        "rlinf.models.embodiment.gr00t.embodiment_tags.EMBODIMENT_TAG_MAPPING",
    )
    Patcher.apply()


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    # Route to N1.6 for behavior_r1_pro (or explicit model_version)
    model_version = getattr(cfg, "model_version", None)
    if model_version == "n1.6" or cfg.embodiment_tag == "behavior_r1_pro":
        return get_model_n1d6(cfg, torch_dtype)
    return get_model_n1d5(cfg, torch_dtype)


def get_model_n1d5(cfg: DictConfig, torch_dtype=torch.bfloat16):
    """Load GR00T N1.5 model (existing code path for libero/maniskill/isaaclab)."""
    from pathlib import Path

    _apply_embodiment_patches()

    from gr00t.experiment.data_config import load_data_config

    from rlinf.models.embodiment.gr00t.gr00t_action_model import (
        GR00T_N1_5_ForRLActionPrediction,
    )
    from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity

    if cfg.embodiment_tag == "libero_franka" or cfg.embodiment_tag == "isaaclab_franka":
        data_config = load_data_config(
            "rlinf.models.embodiment.gr00t.modality_config:LiberoFrankaDataConfig"
        )
    elif cfg.embodiment_tag == "maniskill_widowx":
        data_config = load_data_config(
            "rlinf.models.embodiment.gr00t.modality_config:ManiskillWidowXDataConfig"
        )
    else:
        raise ValueError(f"Invalid embodiment tag for N1.5: {cfg.embodiment_tag}")
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    model = GR00T_N1_5_ForRLActionPrediction.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        embodiment_tag=cfg.embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=cfg.denoising_steps,
        output_action_chunks=cfg.num_action_chunks,
        obs_converter_type=cfg.obs_converter_type,
        tune_visual=False,
        tune_llm=False,
        rl_head_config=cfg.rl_head_config,
    )
    model.to(torch_dtype)
    if cfg.rl_head_config.add_value_head:
        model.action_head.value_head._init_weights()

    if cfg.rl_head_config.disable_dropout:
        replace_dropout_with_identity(model)

    return model


def get_model_n1d6(cfg: DictConfig, torch_dtype=torch.bfloat16):
    """Load GR00T N1.6 model for BEHAVIOR (and future N1.6 environments)."""
    from pathlib import Path

    _apply_embodiment_patches()

    # N1.6 model registration
    import gr00t.model  # noqa: F401

    from transformers import AutoModel, AutoProcessor

    from rlinf.models.embodiment.gr00t.gr00t_n1d6_action_model import (
        GR00T_N1_6_ForRLActionPrediction,
    )
    from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity

    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # Load processor for N1.6 (handles normalization, VLM processing, action decoding)
    processor = AutoProcessor.from_pretrained(model_path)
    processor.eval()

    model = GR00T_N1_6_ForRLActionPrediction.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        embodiment_tag=cfg.embodiment_tag,
        processor=processor,
        denoising_steps=cfg.denoising_steps,
        output_action_chunks=cfg.num_action_chunks,
        obs_converter_type=cfg.obs_converter_type,
        tune_visual=False,
        tune_llm=False,
        rl_head_config=cfg.rl_head_config,
    )
    model.to(torch_dtype)
    if cfg.rl_head_config.add_value_head:
        model.action_head.value_head._init_weights()

    if cfg.rl_head_config.disable_dropout:
        replace_dropout_with_identity(model)

    return model
