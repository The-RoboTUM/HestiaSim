# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration of the kitchen beans-pour task."""

import gymnasium as gym

from . import agents

gym.register(
    id="HestiaBeansPour-Franka-Kitchen-Pour",
    entry_point="hestia_beans_pour.tasks.kitchen_pour.kitchen_env:FrankaKitchenPourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hestia_beans_pour.tasks.kitchen_pour.kitchen_env_cfg:FrankaKitchenPourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaKitchenPourPPORunnerCfg",
    },
)
