# Hestia simulation

Simulation of the Hestia kitchen (roboTUM) in Isaac Lab/Sim, with quick MuJoCo previews on a laptop.
More about the project: https://www.robotum.info/projects/hestia

## Layout

```
HestiaSim/
├── components/        one real object per folder (geometry, drawings, generator scripts, README)
│   ├── aluprofil_table/   lab table from 80x80 aluminium profiles (generator + previews)
│   ├── container/         container01 with handle (STL; script to simplify heavy meshes)
│   ├── franka_panda/      Panda settings used by the pouring training (no robot files)
│   ├── mixer/             mum8 kitchen mixer with chute (STL, IGES, USDA)
│   └── panda_adapter/     round plate between table top and Panda (generator)
├── scenes/            layouts: which components stand where
│   └── u_table_layout/    three tables in a U + mixer + adapter + Panda + container
└── training/          Isaac Lab task packages; each one defines its simulation environment(s)
    └── hestia_beans_pour/ HestiaBeansPour-Franka-Pour: the Panda pours beans between two cups
```

- **Component:** one real object (table, mixer, container).
- **Scene:** a layout, which components stand where. No task, no robot logic.
- **Simulation environment ("env" in Isaac Lab):** a scene plus a task: what the robot can do, what it
  observes, its reward, when an episode resets. Currently each env is defined in its training package,
  e.g. `training/hestia_beans_pour/src/hestia_beans_pour/tasks/pour/pour_env_cfg.py` (no separate
  `sim_env/` folder; open decision, see below).

Every component and scene folder has its own README with the details.

## Rules

1. **Every component and scene folder has a `README.md`.** Add or update it in the same change as the
   files (new object, new version, new units, new position).
2. **Scenes reference components, never copy them**, so a new component version shows up in every scene.
3. **Units:** STL/IGES in **mm** (hand-off files for CAD and hardware); USD, MuJoCo, Isaac in **m**, Z-up.
4. **File names:** `<part>_v<N>.<ext>`, e.g. `container01_v2.stl`. The version counts up per object,
   whoever made it. Keep older versions; the README says which one is current.
5. **Files over 100 MB can't be pushed** (GitHub's limit; it warns from 50 MB). Make a simplified copy with
   `components/container/scripts/simplify_mesh.py` and check that it keeps the physics: same size, volume
   (= mass), centre of mass and inertia, so object handling in Isaac Lab stays realistic. If the full original
   is needed, keep it outside git (e.g. a shared Google Drive, not set up yet) and link it in the README.

## Component README

```markdown
# <Name>

<What it is, one sentence.>

- Made by: <who made each version, e.g. "v1 Hanzhang, v2 Mana", or a link; "unknown" if not known>
- Current file: `<file>` (<units>)
- Origin and front: <e.g. centre of the bottom face; +x = handle side>
- Size: <x × y × z mm> (optional)
- Changes made here: <"none, as received", or what was converted>
```

These lines are the minimum; add details below them when useful (measurements, settings, how to regenerate).
With more than one version, replace "Made by", "Current file" and "Changes made here" with a version table
(see `components/mixer/README.md`):

```markdown
| Version | File | Made by | What changed |
|---|---|---|---|
| v1 | `<part>_v1.stl` (mm) | <name> | <first model> |
| **v2 (current)** | `<part>_v2.stl` (mm) | <name> | <what is different> |
```

## Scene README

A sketch seen from above with the compass, one row per object (component + version, x, y, yaw, what it
stands on), the run commands and what is not done yet. Positions live in one Python file in the scene
folder; the preview and spawn scripts both read it.

## Running

- **Laptop (MuJoCo preview):** `components/` and `scenes/` are enough (sparse checkout works); the exact
  `uv run ...` command is at the top of each script and in its README.
- **Isaac Lab workstation:** the `spawn_*.py` scripts and `training/`; convert the generated meshes to
  USD once (`components/aluprofil_table/scripts/README.md`).

## Open decisions

**Where do simulation environments live? For Jakub (training) to decide.**
Now: inside the training package (`training/hestia_beans_pour/.../tasks/pour/`), which is Isaac Lab's
standard layout: one task package holds the env, registers it under a name
(`HestiaBeansPour-Franka-Pour`) and holds the training configs for it.

| Option | Good when | Cost |
|---|---|---|
| **A. Keep envs inside each training package** (now) | one training approach per env | none |
| **B. Separate `sim_env/` package**, imported by the training packages | several training packages (e.g. RL and imitation learning), evaluation or demos use the same env | one more package; split the current one into env and training |

Suggestion: keep A for now and switch to B once a second user of the same env appears. Either way, the
env should take its table, mixer, robot and positions from `components/` and `scenes/` instead of its own
copies. The kitchen pour env (`tasks/kitchen_pour`, 2026-11-23) already does this through
`scenes/u_table_layout/u_layout.py`; the older pour env still uses its own table and cups at the origin.

**Container position (later).** The container will be spawned at random positions. For now it stands at
one fixed spot, written twice with the same values: `CONTAINER_*` in `u_layout.py` and `CUP_CENTRE_*` in
`kitchen_pour/layout.py` (centre 15 cm from the bottom table's end at the left table, 20 cm from its front
edge). When the random spawn comes, define its range in one place.

AI agents: see `AGENTS.md`.
