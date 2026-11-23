# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configuration of the kitchen pour task."""

from __future__ import annotations

from isaaclab.utils import configclass

from hestia_beans_pour.tasks.pour.mdp.actions_cfg import EMARelativeJointPositionActionCfg


@configclass
class NoisyEMARelativeJointPositionActionCfg(EMARelativeJointPositionActionCfg):
    """:class:`EMARelativeJointPositionActionCfg` of the pour task plus Gaussian actuation noise."""

    noise_std: float = 0.0
    """Standard deviation of the noise added to every commanded joint delta [rad] (0 = off)."""

    class_type: type | str = "hestia_beans_pour.tasks.kitchen_pour.mdp.actions:NoisyEMARelativeJointPositionAction"
