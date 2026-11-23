# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Arm action of the kitchen pour task: the pour task's filtered joint deltas plus actuation noise."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from hestia_beans_pour.tasks.pour.mdp.actions import EMARelativeJointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import NoisyEMARelativeJointPositionActionCfg


class NoisyEMARelativeJointPositionAction(EMARelativeJointPositionAction):
    """Filtered relative joint-position deltas with Gaussian noise on the applied delta.

    The noise is added after the moving-average filter and is not fed back into the filter state, so it models
    joint tracking error of the real controller without changing what the policy's actions mean.
    """

    cfg: NoisyEMARelativeJointPositionActionCfg

    def __init__(self, cfg: NoisyEMARelativeJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._noise_std = float(cfg.noise_std)
        if self._noise_std < 0.0:
            raise ValueError(f"noise_std must be nonnegative, got {self._noise_std}.")

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        if self._noise_std > 0.0:
            self._processed_actions.add_(torch.randn_like(self._processed_actions) * self._noise_std)
