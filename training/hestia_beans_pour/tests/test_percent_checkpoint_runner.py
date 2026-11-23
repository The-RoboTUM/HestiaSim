# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Percent checkpoints of the kitchen pour runner (runs without Isaac Sim and without rsl_rl).

The runner module is loaded by path on top of a stand-in ``rsl_rl.runners.OnPolicyRunner`` whose ``learn()`` follows
the loop of rsl-rl-lib 5.5.1: per iteration ``current_learning_iteration = it``, ``logger.log(it=...)``, a save every
``save_interval`` iterations, and a final save.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.kitless]

RUNNER_PY = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "hestia_beans_pour"
    / "tasks"
    / "kitchen_pour"
    / "config"
    / "franka"
    / "agents"
    / "percent_checkpoint_runner.py"
)


class _Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = object() if log_dir is not None else None
        self.files_seen = []  # checkpoint files present at the start of every iteration's logging

    def log(self, **kwargs):
        if self.log_dir is not None:
            self.files_seen.append(sorted(os.listdir(self.log_dir)))


class _OnPolicyRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.logger = _Logger(log_dir)
        self.current_learning_iteration = 0
        self.saved = []

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            self.current_learning_iteration = it
            self.logger.log(it=it, start_it=start_it, total_it=total_it)
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path, infos=None):
        Path(path).write_text(str(self.current_learning_iteration))
        self.saved.append((os.path.basename(path), self.current_learning_iteration))


@pytest.fixture
def runner_module(monkeypatch):
    rsl_rl = types.ModuleType("rsl_rl")
    runners = types.ModuleType("rsl_rl.runners")
    runners.OnPolicyRunner = _OnPolicyRunner
    rsl_rl.runners = runners
    monkeypatch.setitem(sys.modules, "rsl_rl", rsl_rl)
    monkeypatch.setitem(sys.modules, "rsl_rl.runners", runners)
    spec = importlib.util.spec_from_file_location("percent_checkpoint_runner_under_test", RUNNER_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner(module, log_dir, step=10):
    cfg = {"save_interval": 25, "checkpoint_percent_step": step}
    return module.PercentCheckpointOnPolicyRunner(None, cfg, log_dir=None if log_dir is None else str(log_dir))


def test_completed_percent(runner_module):
    cp = runner_module.completed_percent
    completed = (0, 1, 799, 800, 801, 1599, 1600, 7999, 8000)
    assert [cp(c, 8000, 10) for c in completed] == [0, 0, 0, 10, 10, 10, 20, 90, 100]
    assert [cp(c, 7, 10) for c in range(1, 8)] == [10, 20, 40, 50, 70, 80, 100]
    assert runner_module.percent_checkpoint_name(20) == "model_20pct.pt"


def test_one_checkpoint_rewritten_every_10_percent(runner_module, tmp_path):
    runner = _runner(runner_module, tmp_path)
    runner.learn(800)
    # saved after 80, 160, ..., 800 completed iterations (iteration index = completed - 1), nothing else
    assert runner.saved == [(f"model_{10 * k}pct.pt", 80 * k - 1) for k in range(1, 11)]
    # never more than one checkpoint in the run folder, and only the 100 % one at the end
    assert all(len(files) <= 1 for files in runner.logger.files_seen)
    assert sorted(os.listdir(tmp_path)) == ["model_100pct.pt"]
    assert (tmp_path / "model_100pct.pt").read_text() == "799"
    # the run folder held 10 %, then 20 %, ... in turn
    seen = [files[0] for files in runner.logger.files_seen if files]
    assert list(dict.fromkeys(seen)) == [f"model_{10 * k}pct.pt" for k in range(1, 10)]


def test_short_and_resumed_training(runner_module, tmp_path):
    runner = _runner(runner_module, tmp_path / "short")
    (tmp_path / "short").mkdir()
    runner.learn(7)
    assert [name for name, _ in runner.saved] == [f"model_{p}pct.pt" for p in (10, 20, 40, 50, 70, 80, 100)]
    assert os.listdir(tmp_path / "short") == ["model_100pct.pt"]

    # resumed from iteration 500: the percentages refer to this run's 100 iterations
    (tmp_path / "resumed").mkdir()
    runner = _runner(runner_module, tmp_path / "resumed", step=25)
    runner.current_learning_iteration = 500
    runner.learn(100)
    assert runner.saved == [
        ("model_25pct.pt", 524),
        ("model_50pct.pt", 549),
        ("model_75pct.pt", 574),
        ("model_100pct.pt", 599),
    ]
    assert os.listdir(tmp_path / "resumed") == ["model_100pct.pt"]


def test_no_log_dir_and_explicit_save(runner_module, tmp_path):
    runner = _runner(runner_module, None)
    runner.learn(20)
    assert runner.saved == []
    # outside of learn(), save() writes the requested file as usual
    runner = _runner(runner_module, tmp_path)
    runner.save(str(tmp_path / "model_manual.pt"))
    assert runner.saved == [("model_manual.pt", 0)]
    # the original logger.log is restored after training
    runner.learn(10)
    assert "log_and_checkpoint" not in getattr(runner.logger.log, "__qualname__", "")


def test_invalid_step(runner_module, tmp_path):
    with pytest.raises(ValueError):
        _runner(runner_module, tmp_path, step=30)
