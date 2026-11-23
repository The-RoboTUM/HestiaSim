# Franka Panda (pouring training robot)

The Franka Panda as the Hestia pouring training uses it, in one place, so the training and the scenes use
the same robot. No robot files are stored here: the model comes with Isaac Lab; this folder only holds the
settings on top of it.

- Made by: values from the pouring training (`training/hestia_beans_pour`, `pour_env_cfg.py` as of
  2026-11-22), collected here by Jeffrey
- Current file: `franka_cfg.py`, settings **v1** (`VERSION`); units rad, m, Nm
- Origin and front: base (link0) origin = joint 1 axis on the base's bottom face; +x = robot front
- Size: (robot model from Isaac Lab, not stored here)
- Changes made here: none; **not final yet**. When the training changes the robot, update `franka_cfg.py`
  and bump `VERSION`. Best: the training imports `isaac_cfg()` from here, so there is only one copy.
  Trained policies should note which version they were trained with.

## Settings v1

| Setting | v1 |
|---|---|
| Base | Isaac Lab `FRANKA_PANDA_CFG`, USD `Robots/FrankaEmika/franka_panda.usda`, variants Physics = physx, Colliders = convex_hulls (Isaac Lab dev checkout, isaaclab 47.0.0) |
| Gravity on the robot | off |
| Arm stiffness (joints 1-2 / 3-4 / 5-7) | 1000 / 750 / 300 |
| Arm damping (joints 1-2 / 3-4 / 5-7) | 20 / 4 / 2 |
| Arm force limit (joints 1-4 / 5-7) | 87 / 12 Nm |
| Start pose (joints 1-7) | -1.075, 0.769, 0.532, -2.932, 2.507, 1.400, 0.211 rad; fingers open 0.04 m |

Not included in v1: the training's own spawn function (extra arm collision proxies) and its PhysX
articulation tweaks; they are training details.

## Use

```python
import sys; sys.path.insert(0, "<HestiaSim>/components/franka_panda")
import franka_cfg

robot_cfg = franka_cfg.isaac_cfg("/World/Robot")   # Isaac Lab ArticulationCfg (needs Isaac Lab)
mj = franka_cfg.mujoco_settings()                  # same values for MuJoCo Menagerie's panda.xml
```

MuJoCo has no copy of Isaac Lab's USD, so the MuJoCo preview applies these settings to MuJoCo Menagerie's
`franka_emika_panda/panda.xml` (built from the same Franka description: same links and joints). Same robot
and servo values; the physics engines differ, so test trained policies in Isaac Lab.

Mounted on `components/panda_adapter` in `scenes/u_table_layout`. The kitchen pour training
(`HestiaBeansPour-Franka-Kitchen-Pour`) uses the pour task's robot config directly (`pour_env_cfg._robot_cfg()`).
That config has the same v1 values plus the training's collision proxies, and the robot is mounted at this scene's
pose.
