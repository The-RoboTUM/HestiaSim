# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the Franka beans-pouring task."""

import gymnasium as gym

from . import agents

gym.register(
    id="HestiaBeansPour-Franka-Pour",
    entry_point="hestia_beans_pour.tasks.pour.pour_env:FrankaBeansPourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hestia_beans_pour.tasks.pour.pour_env_cfg:FrankaBeansPourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaBeansPourPPORunnerCfg",
    },
)
