# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms of the kitchen beans-pour task.

Isaac Lab's standard terms, the pour task's terms that are reused unchanged (actions, observations, terminations,
terminal rewards), and the kitchen-specific terms.
"""

from isaaclab.envs.mdp import *  # noqa: F403

from hestia_beans_pour.tasks.pour.mdp.actions_cfg import CurriculumGripperPositionActionCfg
from hestia_beans_pour.tasks.pour.mdp.observations import (
    finger_position_obs,
    finger_velocity_obs,
    gripper_contact_obs,
    gripper_target_obs,
    normalized_cup_velocity_obs,
    time_remaining_obs,
)
from hestia_beans_pour.tasks.pour.mdp.rewards import pour_success_bonus, terminal_failure
from hestia_beans_pour.tasks.pour.mdp.terminations import (
    bean_out_of_bounds,
    excessive_spill,
    extreme_rigid_state,
    immediate_pour_success,
    nonfinite_failure,
    unsuccessful_time_out,
)

from .actions_cfg import NoisyEMARelativeJointPositionActionCfg
from .events import reset_kitchen_scene
from .observations import (
    bean_fractions_obs,
    cup_pose_b_obs,
    cup_tilt_obs,
    grasp_to_tcp_quat_obs,
    mixer_pose_b_obs,
    mouth_to_target_b_obs,
    task_stage_obs,
    tcp_pose_b_obs,
    tcp_to_grasp_c_obs,
)
from .rewards import DeltaReward, grasp_quality, kitchen_pour_potential_terms
from .terminations import cup_dropped
