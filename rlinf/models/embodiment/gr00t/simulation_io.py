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

import numpy as np
import torch
import torch.nn.functional as F


def convert_libero_obs_to_gr00t_format(env_obs):
    """
    Convert the observation to the format expected by the GR00T model.
    The data format is determined by the modality_config and meta/info.json following LeRobot format.
    Considering that we don't have a unified data inferface, we use direct logic here.
    """
    groot_obs = {}

    # [B, H, W, C] -> [B, T, H, W, C]
    groot_obs["video.image"] = env_obs["main_images"].unsqueeze(1).numpy()
    groot_obs["video.wrist_image"] = env_obs["wrist_images"].unsqueeze(1).numpy()
    # [B, 8] -> [B, T(1), 8]
    groot_obs["state.x"] = env_obs["states"].unsqueeze(1)[:, :, 0:1].numpy()
    groot_obs["state.y"] = env_obs["states"].unsqueeze(1)[:, :, 1:2].numpy()
    groot_obs["state.z"] = env_obs["states"].unsqueeze(1)[:, :, 2:3].numpy()
    groot_obs["state.roll"] = env_obs["states"].unsqueeze(1)[:, :, 3:4].numpy()
    groot_obs["state.pitch"] = env_obs["states"].unsqueeze(1)[:, :, 4:5].numpy()
    groot_obs["state.yaw"] = env_obs["states"].unsqueeze(1)[:, :, 5:6].numpy()
    groot_obs["state.gripper"] = env_obs["states"].unsqueeze(1)[:, :, 6:].numpy()
    groot_obs["annotation.human.action.task_description"] = env_obs["task_descriptions"]

    return groot_obs


def convert_maniskill_obs_to_gr00t_format(env_obs):
    """
    Convert the observation to the format expected by the GR00T model.
    The data format is determined by the modality_config and meta/info.json following LeRobot format.
    Considering that we don't have a unified data inferface, we use direct logic here.
    """
    groot_obs = {}
    # video
    # TODO(lx): If we have a dataset on maniskill, resize can be avoided.
    # But now we have to resize images to libero data version.
    env_obs["main_images"] = cut_and_resize_images(
        env_obs["main_images"],
        env_obs["main_images"].shape[-3],  # H
        256,
    )
    # [B, H, W, C] -> [B, T, H, W, C]
    groot_obs["video.ego_view"] = env_obs["main_images"].unsqueeze(1).numpy()
    # state
    if "state" in env_obs:
        raise NotImplementedError("State from simulation are not unified yet.")
    else:
        # gr00t pad the state to input dimension
        # create state of [B, T, C]
        groot_obs["state.left_arm"] = np.zeros((env_obs["main_images"].shape[0], 1, 7))
    # annotation
    groot_obs["annotation.human.action.task_description"] = env_obs["task_descriptions"]
    return groot_obs


def convert_to_libero_action(
    action_chunk: dict[str, np.array], chunk_size: int = 1
) -> np.ndarray:
    """Convert GR00T action chunk to Libero format.

    Args:
        action_chunk: Dictionary of action components from GR00T policy
        chunk_size: Number of action steps to consider from the chunk

    Returns:
        7-dim numpy array: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    action_components = [
        action_chunk["action.x"][:, :chunk_size],
        action_chunk["action.y"][:, :chunk_size],
        action_chunk["action.z"][:, :chunk_size],
        action_chunk["action.roll"][:, :chunk_size],
        action_chunk["action.pitch"][:, :chunk_size],
        action_chunk["action.yaw"][:, :chunk_size],
        action_chunk["action.gripper"][:, :chunk_size],
    ]
    action_array = np.concatenate(action_components, axis=-1)
    action_array = normalize_gripper_action(action_array, binarize=True)
    assert action_array.shape[-1] == 7, (
        f"Expected 7-dim action, got {action_array.shape[-1]}"
    )
    return action_array


def convert_to_maniskill_action(
    action_chunk: dict[str, np.array], chunk_size: int = 16
) -> np.ndarray:
    """Convert GR00T action chunk to Maniskill format."""
    # Accord to gr1 definition, action.left_arm happens to be 7 dims, matching the demand of maniskill.

    return action_chunk["action.left_arm"][:, :chunk_size]


def convert_to_isaaclab_stack_cube_action(
    action_chunk: dict[str, np.array], chunk_size: int = 1
) -> np.ndarray:
    """Convert GR00T action chunk to Isaaclab Stack Cube format.
    The main difference of Libero and Isaaclab Stack Cube is gripper action in
    Libero is 0 and 1, but in Isaaclab Stack Cube is -1 and +1.

    Args:
        action_chunk: Dictionary of action components from GR00T policy
        chunk_size: Number of action steps to consider from the chunk

    Returns:
        7-dim numpy array: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    action_components = [
        action_chunk["action.x"][:, :chunk_size],
        action_chunk["action.y"][:, :chunk_size],
        action_chunk["action.z"][:, :chunk_size],
        action_chunk["action.roll"][:, :chunk_size],
        action_chunk["action.pitch"][:, :chunk_size],
        action_chunk["action.yaw"][:, :chunk_size],
        action_chunk["action.gripper"][:, :chunk_size],
    ]
    action_array = np.concatenate(action_components, axis=-1)
    action_array[..., -1] = np.sign(action_array[..., -1])
    assert action_array.shape[-1] == 7, (
        f"Expected 7-dim action, got {action_array.shape[-1]}"
    )
    return action_array


# Proprioception indices for BEHAVIOR R1Pro (258D proprio → named state keys).
# Only the 22 keys used by the model (matching embodiment_configs.py behavior_r1_pro).
BEHAVIOR_PROPRIO_INDICES = {
    "robot_pos": np.s_[140:143],  # 3
    "robot_ori_cos": np.s_[143:146],  # 3
    "robot_ori_sin": np.s_[146:149],  # 3
    "robot_2d_ori": np.s_[149:150],  # 1
    "robot_2d_ori_cos": np.s_[150:151],  # 1
    "robot_2d_ori_sin": np.s_[151:152],  # 1
    "robot_lin_vel": np.s_[152:155],  # 3
    "robot_ang_vel": np.s_[155:158],  # 3
    "arm_left_qpos": np.s_[158:165],  # 7
    "arm_left_qpos_sin": np.s_[165:172],  # 7
    "arm_left_qpos_cos": np.s_[172:179],  # 7
    "eef_left_pos": np.s_[186:189],  # 3
    "eef_left_quat": np.s_[189:193],  # 4
    "gripper_left_qpos": np.s_[194:196],  # 2
    "arm_right_qpos": np.s_[198:205],  # 7
    "arm_right_qpos_sin": np.s_[205:212],  # 7
    "arm_right_qpos_cos": np.s_[212:219],  # 7
    "eef_right_pos": np.s_[226:229],  # 3
    "eef_right_quat": np.s_[229:233],  # 4
    "gripper_right_qpos": np.s_[234:236],  # 2
    "trunk_qpos": np.s_[238:242],  # 4
}  # total dim = 82

# Action layout for BEHAVIOR R1Pro (23D action vector).
BEHAVIOR_ACTION_MAP = {
    "base": np.s_[0:3],
    "torso": np.s_[3:7],
    "left_arm": np.s_[7:14],
    "left_gripper": np.s_[14:15],
    "right_arm": np.s_[15:22],
    "right_gripper": np.s_[22:23],
}


def convert_behavior_obs_to_gr00t_format(env_obs):
    """
    Convert BEHAVIOR R1Pro observation to the format expected by GR00T N1.6.

    env_obs keys (from behavior_env.py _wrap_obs):
      - main_images: [B, H, W, C] — head/zed camera
      - wrist_images: [B, 2, H, W, C] — [left_wrist, right_wrist]
      - states: [B, 258] — full R1Pro proprioception
      - task_descriptions: list[str] of length B
    """
    groot_obs = {}

    # Video: [B, H, W, C] -> [B, T=1, H, W, C]
    groot_obs["video.observation.images.rgb.head_256_256"] = (
        env_obs["main_images"].unsqueeze(1).numpy()
    )
    groot_obs["video.observation.images.rgb.left_wrist_256_256"] = (
        env_obs["wrist_images"][:, 0].unsqueeze(1).numpy()
    )
    groot_obs["video.observation.images.rgb.right_wrist_256_256"] = (
        env_obs["wrist_images"][:, 1].unsqueeze(1).numpy()
    )

    # State: split 258D proprio into named keys, each [B, T=1, D]
    states = env_obs["states"]
    for key, slc in BEHAVIOR_PROPRIO_INDICES.items():
        groot_obs[f"state.{key}"] = states[:, slc].unsqueeze(1).numpy()

    # Language
    groot_obs["annotation.human.coarse_action"] = env_obs["task_descriptions"]

    return groot_obs


def convert_to_behavior_action(
    action_chunk: dict[str, np.array], chunk_size: int = 32
) -> np.ndarray:
    """Convert GR00T action chunk dict to BEHAVIOR 23D action tensor.

    Args:
        action_chunk: Dictionary of action components from GR00T policy.
        chunk_size: Number of action steps to use from the chunk.

    Returns:
        np.ndarray of shape [B, chunk_size, 23].
    """
    batch_size = action_chunk["action.base"].shape[0]
    action_array = np.zeros(
        (batch_size, chunk_size, 23), dtype=np.float32
    )
    for key, slc in BEHAVIOR_ACTION_MAP.items():
        action_array[..., slc] = action_chunk[f"action.{key}"][:, :chunk_size]
    return action_array


# TODO: we need a unified embodiement data.
OBS_CONVERSION = {
    "maniskill": convert_maniskill_obs_to_gr00t_format,
    "libero": convert_libero_obs_to_gr00t_format,
    "isaaclab_stack_cube": convert_libero_obs_to_gr00t_format,
    "behavior": convert_behavior_obs_to_gr00t_format,
}

ACTION_CONVERSION = {
    "libero": convert_to_libero_action,
    "maniskill": convert_to_maniskill_action,
    "isaaclab_stack_cube": convert_to_isaaclab_stack_cube_action,
    "behavior": convert_to_behavior_action,
}


def cut_and_resize_images(
    images: torch.Tensor, crop_size: int, target_size: int = 256
) -> torch.Tensor:
    """
    Cut and resize the images to the crop size.
    """
    images_nchw = images.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

    original_width = images_nchw.shape[-1]  # W
    start = (original_width - crop_size) // 2
    end = start + crop_size

    # Crop: keep batch, channels, full height; crop width to [start:end]
    cropped_tensor = images_nchw[:, :, :, start:end]  # [B, C, H, crop_W]

    # Resize: interpolate to target_size x target_size
    resized_tensor = F.interpolate(
        cropped_tensor,
        size=(target_size, target_size),
        mode="bilinear",  # Or 'bicubic' for smoother results
        align_corners=False,
    )  # [B, C, target_size, target_size]

    # Convert back to NHWC
    resized_nhwc = resized_tensor.permute(
        0, 2, 3, 1
    ).contiguous()  # [B, C, H, W] -> [B, H, W, C]
    return resized_nhwc


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [+1,-1].

    Normalization formula: y = 1 - 2 * (x - orig_low) / (orig_high - orig_low)
    """
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 1 - 2 * (action[..., -1] - orig_low) / (orig_high - orig_low)

    if binarize:
        action[..., -1] = np.sign(action[..., -1])

    return action
