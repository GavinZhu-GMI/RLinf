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
Here we define the Modality config not present in the gr00t package.
The modality config of gr00t is in gr00t/experiment/data_config.py
"""

from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.experiment.data_config import BaseDataConfig
from gr00t.model.transforms import GR00TTransform


class ManiskillWidowXDataConfig(BaseDataConfig):
    """
    Modality config for Maniskill WidowX.
    """

    video_keys = ["video.ego_view"]
    state_keys = ["state.left_arm"]
    action_keys = ["action.left_arm"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(
                apply_to=self.video_keys, height=224, width=224, interpolation="linear"
            ),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=dict.fromkeys(self.action_keys, "min_max"),
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class BehaviorR1ProDataConfig(BaseDataConfig):
    """
    Modality config for BEHAVIOR R1Pro.
    Matches Isaac-GR00T embodiment_configs.py behavior_r1_pro definition.
    """

    video_keys = [
        "video.observation.images.rgb.head_256_256",
        "video.observation.images.rgb.left_wrist_256_256",
        "video.observation.images.rgb.right_wrist_256_256",
    ]
    state_keys = [
        "state.robot_pos",  # 3
        "state.robot_ori_cos",  # 3
        "state.robot_ori_sin",  # 3
        "state.robot_2d_ori",  # 1
        "state.robot_2d_ori_cos",  # 1
        "state.robot_2d_ori_sin",  # 1
        "state.robot_lin_vel",  # 3
        "state.robot_ang_vel",  # 3
        "state.arm_left_qpos",  # 7
        "state.arm_left_qpos_sin",  # 7
        "state.arm_left_qpos_cos",  # 7
        "state.eef_left_pos",  # 3
        "state.eef_left_quat",  # 4
        "state.gripper_left_qpos",  # 2
        "state.arm_right_qpos",  # 7
        "state.arm_right_qpos_sin",  # 7
        "state.arm_right_qpos_cos",  # 7
        "state.eef_right_pos",  # 3
        "state.eef_right_quat",  # 4
        "state.gripper_right_qpos",  # 2
        "state.trunk_qpos",  # 4
    ]  # total dim = 82
    action_keys = [
        "action.base",  # 3
        "action.torso",  # 4
        "action.left_arm",  # 7
        "action.left_gripper",  # 1
        "action.right_arm",  # 7
        "action.right_gripper",  # 1
    ]  # total dim = 23
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    action_indices = list(range(32))  # BEHAVIOR uses 32-step action horizon

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(
                apply_to=self.video_keys, height=256, width=256, interpolation="linear"
            ),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=dict.fromkeys(self.state_keys, "min_max"),
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=dict.fromkeys(self.action_keys, "min_max"),
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=29,
                max_action_dim=29,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class LiberoFrankaDataConfig(BaseDataConfig):
    video_keys = [
        "video.image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self, action_norm: str = "min_max") -> ModalityTransform:
        if action_norm == "min_max":
            action_transform = StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=dict.fromkeys(self.action_keys, "min_max"),
            )
        else:
            action_transform = StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "mean_std",
                    "action.y": "mean_std",
                    "action.z": "mean_std",
                    "action.roll": "mean_std",
                    "action.pitch": "mean_std",
                    "action.yaw": "mean_std",
                    "action.gripper": "min_max",
                },
            )
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(
                apply_to=self.video_keys, height=224, width=224, interpolation="linear"
            ),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=dict.fromkeys(self.state_keys, "min_max"),
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            action_transform,
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
