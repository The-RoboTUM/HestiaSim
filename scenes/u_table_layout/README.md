# U table layout

Three aluprofil tables (type `no_brace`, no diagonal braces) in a U, with the mixer, the Panda adapter
plate and a Franka Panda on the left table and the container on the bottom table. This scene only
references the components; it never copies their geometry.

## Layout

```
Left / right = as seen standing in the opening of the U, looking at the bottom table.

              north (-y)
       ┌──────────────────┐
 ┌────┐│ C   TableBottom  │┌────┐   back walls outside, fronts facing the inside;
 │ A  │└──────────────────┘│    │   side tables outside the ends of the bottom table,
 │ L  │ M─►    inside      │ R  │   their back ends halfway along the bottom table's depth
 │    │                    │    │
 └────┘                    └────┘   M = mixer, middle of the left table, front (bowl) to the inside
 west (+x)  ▲ you stand here  east (-x)
                                    A = Panda adapter + Panda, left table at its end next to the
                                        bottom table, robot facing 45° "south-east"
                                    C = container, bottom table at its end next to the robot,
                                        handle pointing west (towards the robot side)
```

Origin = centre of the bottom table on the floor, metres, Z-up. Compass as in the viewer's start view:
north = towards the bottom table (-y), south = towards the opening (+y), east = -x, west = +x.
(Correction 2026-11-23: earlier sketches drew the left and right tables mirrored; the positions were
always right.)
Overall 3720 x 2280 mm, inside opening 1800 mm.

## Positions

| Object (component, version) | x [m] | y [m] | yaw | stands on |
|---|---|---|---|---|
| TableBottom (`aluprofil_table` v1, no_brace) | 0 | 0 | 0° | floor |
| TableLeft (same) | +1.38 | +0.90 | +90° | floor |
| TableRight (same) | -1.38 | +0.90 | -90° | floor |
| Mixer (`mixer` v2, Mana) | +1.25 | +0.90 | 180° (front to the inside of the U) | left table top (z 0.68) |
| Panda adapter (`panda_adapter` v1) | +1.175 | +0.250 | -135° (robot faces +135°, "south-east") | left table top; edge 75 mm from its end next to the bottom table, 100 mm from its front edge |
| Franka Panda (`franka_panda` v1) | +1.181 | +0.244 | +135° ("south-east") | adapter (z 0.69), upright home pose |
| Container (`container` v2) | +0.655 | +0.320 | 0° (handle west) | bottom table top (z 0.68); handle tip 150 mm from its end next to the robot, body 100 mm from its front edge, 0.53 m from the robot axis |

All positions live in `u_layout.py` (`placements()`, `MIXER_ON`, `ADAPTER_*`, `ROBOT_FACING_DEG`,
`CONTAINER_*`); the preview and the spawn script both read it.

## Run

| File | Does |
|---|---|
| `u_layout.py` | all positions (pure Python, used by both scripts) |
| `preview_U_table_layout.py` | MuJoCo preview on a laptop (mixer and container visual only, no collision) |
| `spawn_U_table_layout.py` | Isaac Lab: table USD x3 + mixer USD + adapter USD + Panda |

```bash
# laptop (from this folder)
UV="uv run --no-project --with mujoco --with trimesh --with shapely --with mapbox-earcut --with scipy --with networkx --with fast-simplification"
$UV python preview_U_table_layout.py                          # viewer
$UV python preview_U_table_layout.py --drop-test --steps 300  # check: 9 test cubes rest at 0.72 m
# options: --gap, --side-offset, --no-mixer, --no-adapter, --no-robot, --no-container

# Isaac Lab workstation (from your IsaacLab folder). Needs the no_brace table USD and the adapter USD once:
# see components/aluprofil_table/scripts/README.md
uv run python <HestiaSim>/scenes/u_table_layout/spawn_U_table_layout.py --visualizer kit
```

## Robot

The Panda's settings come from `components/franka_panda` (v1 = the pouring training's robot). In this
scene it starts in the upright **home** pose (0, 0, 0, -1.571, 0, 1.571, -0.785 rad; `ROBOT_SCENE_POSE`),
not the training's start pose.
- **Isaac Lab:** `franka_cfg.isaac_cfg()`.
- **MuJoCo preview:** the same settings on MuJoCo Menagerie's `franka_emika_panda/panda.xml` (not in this
  repo). Default path `~/Projects/RoboTUM/franka-sandbox/third_party/mujoco_menagerie/...`; set
  `HESTIA_PANDA_MJCF` or pass `--panda-xml`. Get it with
  `git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git`
  then `git sparse-checkout set franka_emika_panda`.

## Not done yet

- Container is in the MuJoCo preview only; Isaac needs a container USD first.
- Container handle points west (square to the table); later it should point at the robot (`CONTAINER_YAW`).
- This is a layout, not a training environment: the pouring training still uses its own table and cups at
  the origin; running a trained policy here needs those relative positions to match.
