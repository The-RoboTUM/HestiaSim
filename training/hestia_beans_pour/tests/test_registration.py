# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the project's task registrations."""

import gymnasium as gym

import hestia_beans_pour.tasks  # noqa: F401


def test_task_registrations():
    """The project tasks must expose valid environment and agent entry points."""
    expected = {
        "HestiaBeansPour-Balance-Cartpole": {
            "entry_point": "hestia_beans_pour.tasks.balance.config.cartpole.env:BalanceEnv",
            "env_cfg_entry_point": "hestia_beans_pour.tasks.balance.config.cartpole.env_cfg:BalanceEnvCfg",
        },
        "HestiaBeansPour-Balance-Cartpole-Direct": {
            "entry_point": "hestia_beans_pour.tasks.balance_direct.config.cartpole.env:BalanceEnv",
            "env_cfg_entry_point": "hestia_beans_pour.tasks.balance_direct.config.cartpole.env_cfg:BalanceEnvCfg",
        },
        "HestiaBeansPour-Balance-Marl-Cartpole-Direct": {
            "entry_point": "hestia_beans_pour.tasks.balance_marl_direct.config.cartpole.env:BalanceMarlEnv",
            "env_cfg_entry_point": "hestia_beans_pour.tasks.balance_marl_direct.config.cartpole.env_cfg:BalanceMarlEnvCfg",
            "default_agent": "skrl",
        },
        "HestiaBeansPour-Franka-Pour": {
            "entry_point": "hestia_beans_pour.tasks.pour.pour_env:FrankaBeansPourEnv",
            "env_cfg_entry_point": "hestia_beans_pour.tasks.pour.pour_env_cfg:FrankaBeansPourEnvCfg",
        },
        "HestiaBeansPour-Franka-Kitchen-Pour": {
            "entry_point": "hestia_beans_pour.tasks.kitchen_pour.kitchen_env:FrankaKitchenPourEnv",
            "env_cfg_entry_point": "hestia_beans_pour.tasks.kitchen_pour.kitchen_env_cfg:FrankaKitchenPourEnvCfg",
        },
    }

    for task_id, expected_values in expected.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == expected_values["entry_point"]
        assert spec.kwargs["env_cfg_entry_point"] == expected_values["env_cfg_entry_point"]
        if "default_agent" in expected_values:
            assert spec.kwargs["default_agent"] == expected_values["default_agent"]
