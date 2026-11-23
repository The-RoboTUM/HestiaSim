# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from hestia_beans_pour.tasks.pour.config.franka.agents.rsl_rl_ppo_cfg import FrankaBeansPourPPORunnerCfg

# resolved by RSL-RL when the runner is built (importing it here would pull in rsl_rl at cfg import)
_PERCENT_CHECKPOINT_RUNNER = f"{__name__.rsplit('.', 1)[0]}.percent_checkpoint_runner:PercentCheckpointOnPolicyRunner"


@configclass
class FrankaKitchenPourPPORunnerCfg(FrankaBeansPourPPORunnerCfg):
    """PPO for the kitchen pour, learned from the home pose (no reset dataset).

    Asymmetric actor-critic: the actor sees only the robot's own measurements (deployable without cameras), the
    critic additionally sees the privileged scene state.

    Checkpoints: one file per run in ``logs/rsl_rl/franka_kitchen_pour/<run>/``, rewritten after every 10 % of
    ``max_iterations`` as ``model_10pct.pt``, ``model_20pct.pt``, ... ``model_100pct.pt`` (the previous one is deleted),
    see :mod:`.percent_checkpoint_runner`.
    """

    class_name = _PERCENT_CHECKPOINT_RUNNER
    checkpoint_percent_step: int = 10
    """Save (and keep only) a checkpoint after every this many percent of the training iterations."""

    # 48 policy steps = 1.6 s at 30 Hz; episodes are 15 s.
    num_steps_per_env = 48
    max_iterations = 8000
    # Episodes always start at home, so spreading the first episode lengths decorrelates the environments.
    init_at_random_ep_len = True
    # not used: PercentCheckpointOnPolicyRunner replaces RSL-RL's periodic model_<iteration>.pt checkpoints
    save_interval = 100
    obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}
    experiment_name = "franka_kitchen_pour"
    run_name = "from_home"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        # mixed raw and privileged quantities: normalize empirically (the normalizer is exported with the policy)
        obs_normalization=True,
        distribution_cfg=FrankaBeansPourPPORunnerCfg.ExplorationDistributionCfg(init_std=0.60, std_type="log"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=1.0e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99 ** (1.0 / 3.0),
        lam=0.95 ** (1.0 / 3.0),
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
