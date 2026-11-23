# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset event of the kitchen pour task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..kitchen_env import FrankaKitchenPourEnv


def reset_kitchen_scene(env: FrankaKitchenPourEnv, env_ids: torch.Tensor | slice) -> None:
    """Robot to (noisy) home, container to its (noisy) spot on the middle table, half full of beans; mixer
    to its (noisy) spot on the left table. The noise ranges are fields of the env cfg."""
    env.reset_kitchen_scene(env_ids)
