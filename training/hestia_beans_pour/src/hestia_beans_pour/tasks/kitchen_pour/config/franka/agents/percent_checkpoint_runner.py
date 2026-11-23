# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL on-policy runner that keeps exactly one checkpoint per run, rewritten every N % of the training.

Isaac Lab's ``isaaclab train --rl_library rsl_rl`` builds the runner from the agent cfg's ``class_name``
(``rsl_rl.utils.resolve_callable``), so selecting this runner needs no change to the train script: set the runner
cfg's ``class_name`` to ``"<this module>:PercentCheckpointOnPolicyRunner"`` (see ``rsl_rl_ppo_cfg.py``).

Behaviour during :meth:`learn` (``checkpoint_percent_step`` = 10 by default):

* after 10 %, 20 %, ..., 100 % of the iterations of this ``learn`` call, the policy is saved as
  ``<log_dir>/model_<percent>pct.pt`` (for example ``model_10pct.pt``, ``model_20pct.pt``) and the previous one is
  deleted afterwards, so the run folder always holds only the latest checkpoint;
* RSL-RL's own checkpoints (``model_<iteration>.pt`` every ``save_interval`` iterations and at the end) are not
  written; the 100 % checkpoint is the final model.

The files keep RSL-RL's format and the ``model_*.pt`` name pattern, so ``play --checkpoint latest`` and resuming
with ``--checkpoint`` work as before. ``save()`` called outside of training (e.g. by tools) saves normally.
"""

from __future__ import annotations

import logging
import os

from rsl_rl.runners import OnPolicyRunner

logger = logging.getLogger(__name__)


def completed_percent(completed_iterations: int, total_iterations: int, step: int) -> int:
    """Largest multiple of ``step`` percent that ``completed_iterations`` of ``total_iterations`` reached (0 = none).

    Milestone ``p`` is reached after ``ceil(p * total / 100)`` iterations, e.g. 10 % of 8000 after 800.
    """
    if total_iterations <= 0 or completed_iterations <= 0:
        return 0
    percent = min(100, (100 * int(completed_iterations)) // int(total_iterations))
    return (percent // int(step)) * int(step)


def percent_checkpoint_name(percent: int) -> str:
    """File name of the checkpoint written at ``percent`` % of the training."""
    return f"model_{int(percent)}pct.pt"


class PercentCheckpointOnPolicyRunner(OnPolicyRunner):
    """:class:`rsl_rl.runners.OnPolicyRunner` that keeps one checkpoint, rewritten every ``checkpoint_percent_step`` %.

    Reads ``checkpoint_percent_step`` (int, divides 100, default 10) from the train cfg.
    """

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        step = int(self.cfg.get("checkpoint_percent_step", 10))
        if not 1 <= step <= 100 or 100 % step != 0:
            raise ValueError(f"checkpoint_percent_step must divide 100 (1, 2, 4, 5, 10, 20, 25, 50, 100), got {step}.")
        self._percent_step = step
        self._in_learn = False
        self._percent_checkpoint: str | None = None

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        start_iteration = self.current_learning_iteration
        total = int(num_learning_iterations)
        saved_percent = 0
        log = self.logger.log

        # The base loop calls logger.log() once per iteration, after the policy update and after setting
        # current_learning_iteration; checkpointing right behind it saves the freshly updated policy.
        def log_and_checkpoint(*args, **kwargs):
            nonlocal saved_percent
            log(*args, **kwargs)
            completed = self.current_learning_iteration - start_iteration + 1
            percent = completed_percent(completed, total, self._percent_step)
            if percent > saved_percent:
                self._save_percent_checkpoint(percent)
                saved_percent = percent

        self.logger.log = log_and_checkpoint
        self._in_learn = True
        try:
            super().learn(num_learning_iterations, init_at_random_ep_len=init_at_random_ep_len)
        finally:
            self._in_learn = False
            self.logger.log = log

    def save(self, path: str, infos: dict | None = None) -> None:
        if self._in_learn:
            # RSL-RL's periodic model_<it>.pt and final save: replaced by the percent checkpoints
            return
        super().save(path, infos)

    def _save_percent_checkpoint(self, percent: int) -> None:
        # only the process that logs (rank 0 with a log dir) writes checkpoints, like RSL-RL itself
        if self.logger.writer is None or self.logger.log_dir is None:
            return
        path = os.path.join(self.logger.log_dir, percent_checkpoint_name(percent))
        super().save(path)
        previous, self._percent_checkpoint = self._percent_checkpoint, path
        # delete the previous checkpoint only after the new one is written
        if previous is not None and previous != path and os.path.isfile(previous):
            os.remove(previous)
        logger.info("Saved checkpoint %s (iteration %d).", path, self.current_learning_iteration)
        print(f"[checkpoint] {percent}% of the training done: saved {path}")
