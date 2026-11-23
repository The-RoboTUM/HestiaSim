# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terminations of the kitchen pour task that the pour task does not have.

The kitchen env exposes the same state helpers as the pour env, so the pour task's ``nonfinite_failure``,
``extreme_rigid_state``, ``bean_out_of_bounds``, ``excessive_spill``, ``immediate_pour_success`` and
``unsuccessful_time_out`` are reused directly (see ``kitchen_env_cfg.TerminationsCfg``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .rewards import grasp_quality

if TYPE_CHECKING:
    from ..kitchen_env import FrankaKitchenPourEnv


def cup_dropped(
    env: FrankaKitchenPourEnv,
    fall_depth: float = 0.05,
    topple_angle_deg: float = 60.0,
    topple_height: float = 0.03,
) -> torch.Tensor:
    """The container fell off the table, or lies toppled on it while not held."""
    height = env.cup_pose_e()[:, 2] - env.table_top
    fell = height < -float(fall_depth)
    toppled = (
        (env.cup_tilt() > math.radians(float(topple_angle_deg)))
        & (height < float(topple_height))
        & (grasp_quality(env) < 0.1)
    )
    return fell | toppled
