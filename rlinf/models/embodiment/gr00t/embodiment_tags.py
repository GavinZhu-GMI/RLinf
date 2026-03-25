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


from enum import Enum


class EmbodimentTag(Enum):
    # === Upstream N1.6 tags (from Isaac-GR00T/gr00t/data/embodiment_tags.py) ===
    ROBOCASA_PANDA_OMRON = "robocasa_panda_omron"
    GR1 = "gr1"
    UNITREE_G1 = "unitree_g1"
    LIBERO_PANDA = "libero_panda"
    OXE_GOOGLE = "oxe_google"
    OXE_WIDOWX = "oxe_widowx"
    OXE_DROID = "oxe_droid"
    BEHAVIOR_R1_PRO = "behavior_r1_pro"
    NEW_EMBODIMENT = "new_embodiment"

    # === RLinf-added tags ===
    AGIBOT_GENIE1 = "agibot_genie1"
    LIBERO_FRANKA = "libero_franka"
    MANISKILL_WIDOWX = "maniskill_widowx"
    ISAACLAB_FRANKA = "isaaclab_franka"


# Embodiment tag string: to projector index in the Action Expert Module
EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.LIBERO_FRANKA.value: 31,
    EmbodimentTag.OXE_DROID.value: 17,
    EmbodimentTag.AGIBOT_GENIE1.value: 26,
    EmbodimentTag.GR1.value: 24,
    EmbodimentTag.MANISKILL_WIDOWX.value: 30,
    EmbodimentTag.ISAACLAB_FRANKA.value: 31,
    EmbodimentTag.BEHAVIOR_R1_PRO.value: 24,
}
