# hestia_beans_pour

An installable downstream Isaac Lab task package with a standard uv `src` layout.

The registered tasks are:
- `HestiaBeansPour-Franka-Pour` — Franka Panda pours rigid beans from one cup into another (Isaac Sim PhysX)
- `HestiaBeansPour-Balance-Cartpole`
- `HestiaBeansPour-Balance-Cartpole-Direct`
- `HestiaBeansPour-Balance-Marl-Cartpole-Direct`

## Franka beans pour

`HestiaBeansPour-Franka-Pour` (in `src/hestia_beans_pour/tasks/pour`) is a rigid-body port of Isaac Lab's
`IsaacContrib-Franka-Pour`. It keeps the same scene: Franka Panda, SeattleLab table, and the source and receiver cups
from the published `franka_pour_cups.usda`. It also keeps the same actions, observations, terminal-sparse reward, and
adaptive reset-dataset curriculum. The differences are:

- **Physics:** Isaac Sim PhysX (`PhysxCfg`, 240 Hz physics, 30 Hz control) instead of Newton MJWarp + MPM.
- **Media:** the granular MPM fluid is replaced by individual rigid beans (one PhysX rigid body each), grouped in a
  `RigidObjectCollection`. With the defaults, 54 spheres of 5 mm radius fill 70% of the source cup.
- **Cup colliders:** the cups USD was authored for Newton (solid grasp proxy, particle-only meshes). At spawn time
  those colliders are replaced by hollow five-box colliders, so beans rest inside both cups. The outer envelope is
  unchanged, so grasp states in the reset dataset still apply.
- **Robot:** the PhysX payload of the same Franka USD with convex-hull colliders. MuJoCo gravity compensation is
  replaced by disabling gravity on the robot links.
- **Reset dataset:** the published 20,000-row Franka Pour reset artifact is reused for its rigid robot and cup states.
  Only those contract fields are validated. Beans are placed on a lattice in the restored source-cup frame.

The task needs Isaac Sim, so always pass the `isaacsim` extra:

```bash
uv sync --extra isaacsim

# Smoke test
uv run --extra isaacsim isaaclab random_agent --task HestiaBeansPour-Franka-Pour --num_envs 16

# Train (rsl_rl) and play
uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task HestiaBeansPour-Franka-Pour \
  --num_envs 1024 --max_iterations 3000
uv run --extra isaacsim isaaclab play --rl_library rsl_rl --task HestiaBeansPour-Franka-Pour \
  --checkpoint latest --num_envs 1 --visualizer kit
```

The beans are configurable from the command line. The bean collection is rebuilt from these values when the
environment is created:

```bash
# 30 elongated capsule beans, 60% of them must reach the receiver
uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task HestiaBeansPour-Franka-Pour \
  env.bean_shape=capsule env.bean_radius=0.004 env.bean_length=0.004 env.num_beans=30 env.pour_target_frac=0.6
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `env.bean_shape` | `sphere` | `sphere` or `capsule` |
| `env.bean_radius` / `env.bean_length` | `0.005` / `0.004` m | radius; capsule cylinder length |
| `env.bean_mass` | `0.0006` kg | mass of one bean |
| `env.bean_friction` / `env.bean_restitution` | `0.5` / `0.05` | contact material |
| `env.bean_fill_level` | `0.70` | lattice fill height as a fraction of the cup cavity |
| `env.num_beans` | `None` | exact bean count (`None` fills up to `bean_fill_level`) |
| `env.pour_target_frac` | `0.70` | fraction of beans that must be in the receiver for success |
| `env.max_spill_fraction` | `0.10` | spilled fraction that terminates the episode |

Each bean is a separate rigid body, so the cost grows with `num_envs × num_beans`. On an RTX 5090, 1024 envs with 54
beans run at about 9,000 env-steps/s.


## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the project environment. The default
environment uses the Newton backend and does not install Isaac Sim:

```bash
uv sync
```

This project's `pyproject.toml` uses editable relative paths to an Isaac Lab source checkout. Update
`[tool.uv.sources]` if either directory moves.

The project forwards every optional extra declared by the Isaac Lab package used to create it, including physics,
rendering, visualizer, RL, teleoperation, and development features. Inspect `pyproject.toml` for the complete list and
pass each required extra to `uv run`:

```bash
# Standalone OV PhysX
uv run --extra ovphysx isaaclab random_agent --task <TASK_NAME> physics=ovphysx

# Isaac Sim with PhysX and RTX rendering
uv run --extra isaacsim isaaclab random_agent --task <TASK_NAME> physics=isaacsim_physx

# Isaac Sim with the Rerun visualizer
uv run --extra isaacsim --extra rerun isaaclab random_agent --task <TASK_NAME> --viz rerun
```

The `all` extra forwards Isaac Lab's `all` selection. The `ov` extra installs both the `ovphysx` and `ovrtx` runtimes;
you can also select either independently. Commit `pyproject.toml` and `uv.lock` so collaborators use the same
environment.

## Run the tasks

Replace the placeholders below with a task listed above and a selected RL library.

```bash
# List this project's environments and their available presets
uv run isaaclab list_envs --show_presets

# Exercise an environment without a trained policy
uv run isaaclab zero_agent --task <TASK_NAME> --num_envs 16
uv run isaaclab random_agent --task <TASK_NAME> --num_envs 16

# Train and play
uv run isaaclab train --rl_library <RL_LIBRARY> --task <TASK_NAME>
uv run isaaclab play --rl_library <RL_LIBRARY> --task <TASK_NAME> --checkpoint latest

# Distributed training
uv run isaaclab train_multigpu --rl_library <RL_LIBRARY> --task <TASK_NAME> --num_gpus 2

# Benchmark startup, runtime, training, or play
uv run isaaclab benchmark runtime --task <TASK_NAME> --num_envs 16 --num_steps 1000
uv run isaaclab benchmark training --rl_library <RL_LIBRARY> --task <TASK_NAME> --max_iterations 10
```

Use `physics=<PRESET>` to select one of the presets shown by `isaaclab list_envs`.

## Project structure

Task families live in `src/hestia_beans_pour/tasks`. Within a manager-based family, task-wide MDP terms live in `mdp`, while
robot-specific scenes, registrations, and agent configurations live in `config/cartpole`. Add another robot by
creating a sibling of `config/cartpole`; add another task by creating a sibling task-family directory.

## Project assets

Keep project-owned USD files and related data under `src/hestia_beans_pour/assets/data`. The asset module exposes a
stable path that works from an editable checkout and an installed wheel:

```python
from hestia_beans_pour.assets import HESTIA_BEANS_POUR_ASSETS_DIR

robot_usd_path = HESTIA_BEANS_POUR_ASSETS_DIR / "robots" / "my_robot.usd"
```

Keep asset configuration in the `hestia_beans_pour.assets` package and pass `str(robot_usd_path)` to configuration fields that
expect a string. USD files are configured for Git LFS by the project's `.gitattributes`; install Git LFS before adding
large binary assets.

## Development

Run the registration test and code-quality checks through the project environment:

```bash
uv run pytest
uv run pre-commit run --all-files
```

The test helpers under `source/isaaclab_tasks/test` in the Isaac Lab repository are not part of the installed
`isaaclab_tasks` package. Keep test fixtures in this project and use public Isaac Lab APIs. If you copy
`env_test_utils.py`, it becomes vendored code whose upstream changes you must track.

To configure VS Code or Cursor, run the `setup_python_env` task or invoke its command directly:

```bash
uv run isaaclab --editor
```

The setup command selects the active interpreter and writes a git-ignored `pyrightconfig.json`. The resulting
configuration inherits the project's checked-in Pyright settings and adds the Isaac Sim extensions, project `src` root,
and any Isaac Lab packages discovered in the active Python environment. This supports both Pylance in VS Code and
basedpyright in Cursor.

In VS Code, use Pylance and select the interpreter that ran the setup command. In Cursor, install the
[basedpyright extension](https://marketplace.visualstudio.com/items?itemName=detachhead.basedpyright) instead of
Pylance, select the same interpreter, and reload the window. Both language servers read `pyrightconfig.json`.

When using the `isaacsim` extra, include it while generating the editor configuration so the command can discover the
Isaac Sim installation:

```bash
uv run --extra isaacsim isaaclab --editor
```

For an Isaac Sim binaries installation that is not available in the project environment, provide its path explicitly:

```bash
# Linux
uv run isaaclab --editor --isaac_path <isaac-sim-path>

# Windows
uv run isaaclab --editor --isaac_path <isaac-sim-path>
```

## Isaac Sim UI extension

Add the project root to the Isaac Sim Extension Manager search paths, refresh, and enable the extension under
`Third Party`. Launch Isaac Sim through the project's `isaacsim` extra so the UI dependencies are available. Kit loads
`src/hestia_beans_pour/ui_extension_example.py` through `config/extension.toml`.

## Troubleshooting

If Pylance or basedpyright cannot resolve modules, confirm that the selected interpreter matches the one used to run the
setup command, then reload the editor window. To add a missing extension or reduce indexing memory, edit the `extraPaths`
array in the root `pyrightconfig.json`; remove simulator extension directories that the project does not use.
