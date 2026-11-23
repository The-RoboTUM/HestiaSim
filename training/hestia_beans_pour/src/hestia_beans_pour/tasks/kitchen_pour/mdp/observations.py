# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observations of the kitchen pour task.

The *actor* only gets what the real robot measures without cameras: arm joint positions and velocities, the two
finger positions, its own gripper command, its last action and the time left. The container and the mixer always
stand at the same place, so the policy does not need to see them. Everything else (container pose, beans, task
stage) is *privileged* and only goes to the critic (asymmetric actor-critic), which is not needed on the robot.

All poses are expressed in the robot base frame (link0), like on the real robot.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

from hestia_beans_pour.tasks.pour.mdp.observations import nearest_grasp_to_tcp_quat

from .rewards import POTENTIAL_TERMS, kitchen_pour_potential_terms

if TYPE_CHECKING:
    from ..kitchen_env import FrankaKitchenPourEnv


def _canonical_pose(pose: torch.Tensor) -> torch.Tensor:
    pose = torch.nan_to_num(pose)
    return torch.cat((pose[:, :3], math_utils.quat_unique(pose[:, 3:7])), dim=-1)


def tcp_pose_b_obs(env: FrankaKitchenPourEnv) -> torch.Tensor:
    """Tool-centre pose in the robot base frame (position [m] + xyzw quaternion)."""
    return _canonical_pose(env.pose_e_to_b(env.tcp_pose_e()))


def cup_pose_b_obs(env: FrankaKitchenPourEnv) -> torch.Tensor:
    """Container pose in the robot base frame."""
    return _canonical_pose(env.pose_e_to_b(env.cup_pose_e()))


def mixer_pose_b_obs(env: FrankaKitchenPourEnv) -> torch.Tensor:
    """Mixer pose in the robot base frame (randomized a little at every reset)."""
    return _canonical_pose(env.pose_e_to_b(env.mixer_pose_e()))


def tcp_to_grasp_c_obs(env: FrankaKitchenPourEnv) -> torch.Tensor:
    """Grasp point minus TCP, in the container frame [m]."""
    cup = env.cup_pose_e()
    offset = env.cup_grasp_point_e() - env.tcp_pose_e()[:, :3]
    return torch.nan_to_num(math_utils.quat_apply_inverse(cup[:, 3:7], offset))


def grasp_to_tcp_quat_obs(env: FrankaKitchenPourEnv) -> torch.Tensor:
    """TCP orientation error to the nearest of the container's four side grasps."""
    return nearest_grasp_to_tcp_quat(env)


def mouth_to_target_b_obs(env: FrankaKitchenPourEnv) -> torch.Tensor:
    """Pour target above the chute minus the container mouth, in the robot base frame [m]."""
    root_quat = env._robot.data.root_link_pose_w.torch[:, 3:7]
    return torch.nan_to_num(math_utils.quat_apply_inverse(root_quat, env.pour_target_e() - env.cup_mouth_e()))


def cup_tilt_obs(env: FrankaKitchenPourEnv) -> torch.Tensor:
    """Container tilt from upright, normalized by pi."""
    return (env.cup_tilt() / math.pi).unsqueeze(-1)


def bean_fractions_obs(env: FrankaKitchenPourEnv) -> torch.Tensor:
    """Fractions of beans in the container, delivered into the mixer and spilled."""
    return env.bean_fractions()


def task_stage_obs(env: FrankaKitchenPourEnv) -> torch.Tensor:
    """The staged task potential's terms (approach, grasp, lift, transport, pour) and their sum."""
    terms = kitchen_pour_potential_terms(env)
    return torch.stack([terms[name] for name in (*POTENTIAL_TERMS, "potential")], dim=-1)
