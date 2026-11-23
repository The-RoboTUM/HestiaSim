# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rewards of the kitchen pour task.

The pour task learns from a published reset dataset (its start states are spread along the whole task). That dataset
was generated for a robot at the origin and two cups on a lab table, so it cannot be used in the kitchen. Here every
episode starts from the robot's home pose, and learning is driven by one bounded, staged *potential*

    approach -> grasp -> lift -> transport the mouth above the chute -> tilt over the chute

whose change per step is the reward (potential-based shaping). Holding a good state earns nothing and going back
costs exactly what was gained, so the shaping cannot be farmed and does not compete with finishing the pour. On top
come the change of the delivered / spilled bean fractions, a terminal success bonus and a terminal failure penalty.

The container (``components/container``) is grasped at its handle from above, fingers across the 20 mm handle. The
transport and pour stages use its *pouring lip* (the lowest point of the rim, ``env.cup_pour_point_e()``), because
the 150 x 110 mm opening is much wider than the chute.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg
from isaaclab.utils import math as math_utils

from hestia_beans_pour.tasks.pour.mdp.rewards import _grasp_width_quality, _open_quality

if TYPE_CHECKING:
    from ..kitchen_env import FrankaKitchenPourEnv

POTENTIAL_TERMS = ("approach", "grasp", "lift", "transport", "pour")

DEFAULT_POTENTIAL_PARAMS = {
    # stage weights (sum to one)
    "weights": {"approach": 0.15, "grasp": 0.25, "lift": 0.15, "transport": 0.25, "pour": 0.20},
    "approach_position_std": 0.10,
    "approach_orientation_std": 0.75,
    "approach_open_hand_fraction": 0.15,
    "grasp_reach": 0.025,
    # full grasp credit once the filtered finger command is below 6 mm (the fingers stop at 10 mm on the handle);
    # same share of the closing stroke (0.04 -> 0) as 24 mm was of the pour task's stroke (0.04 -> 0.021)
    "grasp_preload_position": 0.006,
    "lift_height": 0.10,
    "transport_min_lift": 0.05,
    "transport_far_distance": 0.80,
    "transport_near_std": 0.08,
    "pour_radius": 0.06,
    "pour_clearance": 0.02,
    # the container is shallow (65 mm cavity): it is empty well before it is upside down
    "pour_tilt_deg": 110.0,
}


def _smoothstep01(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return x.square() * (3.0 - 2.0 * x)


def handle_grasp_to_tcp_quat(env: FrankaKitchenPourEnv) -> torch.Tensor:
    """TCP orientation error ``(x, y, z, w)`` to the nearer of the two equivalent handle grasps.

    The handle grasp (``layout.HANDLE_GRASP_TCP_QUAT``) and the same grasp turned by 180 deg about the TCP axis
    (fingers swapped) hold the handle identically. Unlike the pour task's cup, the container cannot be grasped from
    its other sides (the gripper does not span the box), so there are only these two.
    """
    cup_quat = env.cup_pose_e()[:, 3:7]
    tcp_quat = env.tcp_pose_e()[:, 3:7]
    n = cup_quat.shape[0]
    base = math_utils.quat_mul(cup_quat, env._desired_grasp_tcp_quat_c)
    # 180 deg about the TCP z axis, applied in the TCP frame
    flip = torch.zeros((n, 4), device=cup_quat.device, dtype=cup_quat.dtype)
    flip[:, 2] = 1.0
    desired = torch.stack((base, math_utils.quat_mul(base, flip)), dim=1)
    errors = math_utils.quat_unique(
        math_utils.quat_mul(math_utils.quat_conjugate(desired), tcp_quat.unsqueeze(1).expand(-1, 2, -1))
    )
    magnitudes = torch.linalg.vector_norm(math_utils.axis_angle_from_quat(errors), dim=-1)
    magnitudes = torch.nan_to_num(magnitudes, nan=torch.inf, posinf=torch.inf, neginf=torch.inf)
    nearest = torch.argmin(magnitudes, dim=1)
    error = torch.gather(errors, dim=1, index=nearest[:, None, None].expand(-1, 1, 4)).squeeze(1)
    return torch.nan_to_num(math_utils.quat_unique(error))


def tcp_handle_grasp_pose_tanh(
    env: FrankaKitchenPourEnv,
    position_std: float = 0.10,
    orientation_std: float = 0.75,
    open_hand_fraction: float = 0.15,
) -> torch.Tensor:
    """Approach quality in ``[0, 1]``: TCP near the handle's grasp point, in the handle grasp orientation, hand open.

    Same shape as the pour task's ``tcp_cup_grasp_pose_tanh`` (broad tanh kernels, open-hand bonus that vanishes at
    the grasp pose), with the handle's two grasps instead of the cup's four side grasps.
    """
    distance = torch.linalg.vector_norm(env.tcp_pose_e()[:, :3] - env.cup_grasp_point_e(), dim=-1)
    position_quality = 1.0 - torch.tanh(distance / float(position_std))
    error = handle_grasp_to_tcp_quat(env)
    orientation_error = torch.linalg.vector_norm(math_utils.axis_angle_from_quat(error), dim=-1)
    orientation_quality = 1.0 - torch.tanh(orientation_error / float(orientation_std))
    pose_quality = position_quality * (0.5 + 0.5 * orientation_quality)
    potential = pose_quality + float(open_hand_fraction) * (1.0 - pose_quality) * _open_quality(env)
    return torch.nan_to_num(potential, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


def grasp_quality(env: FrankaKitchenPourEnv, reach: float = 0.025, preload_position: float = 0.006) -> torch.Tensor:
    """Contact-qualified grasp in ``[0, 1]``: TCP at the grasp point, hand commanded closed, both fingers pressing."""
    distance = torch.linalg.vector_norm(env.tcp_pose_e()[:, :3] - env.cup_grasp_point_e(), dim=-1)
    proximity = _smoothstep01(1.0 - distance / float(reach))
    return torch.nan_to_num(proximity * _grasp_width_quality(env, preload_position)).clamp_(0.0, 1.0)


def kitchen_pour_potential_terms(env: FrankaKitchenPourEnv, **params) -> dict[str, torch.Tensor]:
    """Stage qualities in ``[0, 1]`` (see the module docstring) and their weighted sum ``"potential"``."""
    p = {**DEFAULT_POTENTIAL_PARAMS, **params}
    approach = tcp_handle_grasp_pose_tanh(
        env,
        position_std=p["approach_position_std"],
        orientation_std=p["approach_orientation_std"],
        open_hand_fraction=p["approach_open_hand_fraction"],
    )
    grasp = grasp_quality(env, p["grasp_reach"], p["grasp_preload_position"])
    height = env.cup_pose_e()[:, 2] - env.table_top
    lift = grasp * (height / float(p["lift_height"])).clamp(0.0, 1.0)

    # pouring lip (= centre of the opening while upright, the lowest rim point when tilted)
    mouth = env.cup_pour_point_e()
    target = env.pour_target_e()
    target_distance = torch.linalg.vector_norm(mouth - target, dim=-1)
    # broad linear part (useful from the table) + sharp part (precise placement over the chute)
    reach = 0.5 * (1.0 - target_distance / float(p["transport_far_distance"])).clamp(0.0, 1.0)
    reach = reach + 0.5 * (1.0 - torch.tanh(target_distance / float(p["transport_near_std"])))
    lifted = (height / float(p["transport_min_lift"])).clamp(0.0, 1.0)
    transport = grasp * lifted * reach

    chute = env.chute_top_e()
    over_chute = _smoothstep01(1.0 - torch.linalg.vector_norm((mouth - chute)[:, :2], dim=-1) / float(p["pour_radius"]))
    clearance = ((mouth[:, 2] - chute[:, 2]) / float(p["pour_clearance"])).clamp(0.0, 1.0)
    tilt = (env.cup_tilt() / math.radians(float(p["pour_tilt_deg"]))).clamp(0.0, 1.0)
    pour = grasp * over_chute * clearance * tilt

    terms = {"approach": approach, "grasp": grasp, "lift": lift, "transport": transport, "pour": pour}
    weights = p["weights"]
    potential = sum(float(weights[name]) * terms[name] for name in POTENTIAL_TERMS)
    terms["potential"] = torch.nan_to_num(potential, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return terms


class DeltaReward(ManagerTermBase):
    """Reward = change of a bounded quantity since the previous step (divided by ``step_dt``, so the term's weight
    is the total reward for raising the quantity by one over the episode).

    ``quantity`` is ``"potential"`` (the staged task potential), ``"delivered"`` or ``"spilled"`` (bean fractions).
    The first step after a reset only records the quantity, so a reset never pays a reward.
    """

    def __init__(self, cfg: RewardTermCfg, env: FrankaKitchenPourEnv):
        super().__init__(cfg, env)
        self._previous = torch.zeros(env.num_envs, device=env.device)
        self._fresh = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        self._fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env: FrankaKitchenPourEnv, quantity: str = "potential", potential_params: dict | None = None):
        if quantity == "potential":
            value = kitchen_pour_potential_terms(env, **(potential_params or {}))["potential"]
        elif quantity == "delivered":
            value = env.bean_fractions()[:, 1]
        elif quantity == "spilled":
            value = env.bean_fractions()[:, 2]
        else:
            raise ValueError(f"Unknown quantity {quantity!r}; expected 'potential', 'delivered' or 'spilled'.")
        value = torch.nan_to_num(value)
        delta = torch.where(self._fresh, torch.zeros_like(value), value - self._previous)
        self._previous.copy_(value)
        self._fresh.zero_()
        return delta / max(float(env.step_dt), 1.0e-6)


__all__ = [
    "DEFAULT_POTENTIAL_PARAMS",
    "DeltaReward",
    "POTENTIAL_TERMS",
    "grasp_quality",
    "handle_grasp_to_tcp_quat",
    "kitchen_pour_potential_terms",
    "tcp_handle_grasp_pose_tanh",
]
