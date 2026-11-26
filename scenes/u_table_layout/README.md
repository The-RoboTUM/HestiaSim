# U table layout

Three aluprofil tables (type `no_brace`, no diagonal braces) in a U, with a wooden board, mixer, Panda
adapter plate, Franka Panda and the container on the left table (container on the board). This scene only
references the components; it never copies their geometry.

## Layout

```
Left / right = as seen standing in the opening of the U, looking at the bottom table.

              north (-y)
       ┌──────────────────┐
 ┌────┐│     TableBottom  │┌────┐   back walls outside, fronts facing the inside;
 │ A  │└──────────────────┘│    │   side tables outside the ends of the bottom table,
 │ L C│ M─►    inside      │ R  │   their back ends halfway along the bottom table's depth
 │    │                    │    │
 └────┘                    └────┘   M = mixer, middle of the left table, front (bowl) to the inside
 west (+x)  ▲ you stand here  east (-x)
                                    A = Panda adapter + Panda, left table at its end next to the
                                        bottom table, robot facing 45° "south-east"
                                    C = container, on the wooden board between the robot and
                                        the mixer, handle pointing at the robot
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
| Wooden board (`wooden_board` v1) | +1.32 | +0.90 | 0° (long side along world Y) | left table top (z 0.68) |
| Mixer (`mixer` v2, Mana) | +1.25 | +0.90 | 180° (front to the inside of the U) | wooden board (z 0.699) |
| Panda adapter (`panda_adapter` v1) | +1.175 | +0.250 | -135° (robot faces +135°, "south-east") | left table top; edge 75 mm from its end next to the bottom table, 100 mm from its front edge |
| Franka Panda (`franka_panda` v1) | +1.181 | +0.244 | +135° ("south-east") | adapter (z 0.69), upright home pose |
| Container (`container` v3, simplified) | +1.320 | +0.635 | -110° (handle points at the robot axis) | wooden board (z 0.699), between the board's robot-side edge and the mixer, 0.415 m from the robot axis; differs from the kitchen pour training's spot on the bottom table |

Positions live in `u_layout.py`; the MuJoCo preview reads `BOARD_ON` and `MIXER_ON`. The Isaac Lab spawn
script still uses the mixer position but does not yet spawn the board.

## Run

| File | Does |
|---|---|
| `u_layout.py` | all positions (pure Python, used by both scripts) |
| `preview_U_table_layout.py` | MuJoCo preview on a laptop (board, mixer and container visual only, no collision) |
| `spawn_U_table_layout.py` | Isaac Lab: table USD x3 + mixer USD + adapter USD + Panda |

```bash
# laptop (from this folder)
UV="uv run --no-project --with mujoco --with trimesh --with shapely --with mapbox-earcut --with scipy --with networkx --with fast-simplification"
$UV python preview_U_table_layout.py                          # viewer
$UV python preview_U_table_layout.py --drop-test --steps 300  # check: 9 test cubes rest at 0.72 m
# options: --gap, --side-offset, --no-board, --no-mixer, --no-adapter, --no-robot, --no-container

# Isaac Lab workstation (from your IsaacLab folder). Needs the no_brace table USD and the adapter USD once:
# see components/aluprofil_table/scripts/README.md
uv run python <HestiaSim>/scenes/u_table_layout/spawn_U_table_layout.py --visualizer kit
```

The wooden board and the corrected mixer height are currently implemented in the MuJoCo preview only;
the Isaac Lab spawn script still places the mixer directly on the table.

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

**Quaternions:** Isaac Lab 3.x uses (x, y, z, w), so `spawn_U_table_layout.py` uses `u_layout.quat_xyzw()`
(fixed 2026-11-23 by Jakub: it used `quat_wxyz()` before, which Isaac Lab read as a 180° roll, i.e. upside-down
tables). The MuJoCo preview keeps (w, x, y, z).

## Used by training

- **Kitchen pour** (`HestiaBeansPour-Franka-Kitchen-Pour`, `training/hestia_beans_pour/.../tasks/kitchen_pour`)
  reads the tables, the robot pose (on the adapter, home pose) and the mixer from `u_layout.py`. It sets the
  bean container (`components/container`, rebuilt from boxes measured on container01 v2) position itself
  (`layout.py`, `CUP_CENTRE_*`): the centre of its 190 x 120 mm footprint (handle included) 15 cm from the bottom
  table's end at the left table and 20 cm from its front edge, handle pointing to that end (see the training
  README for all edge distances). `CONTAINER_*` here uses the same values, keep them equal.
- **Pour** (`HestiaBeansPour-Franka-Pour`) does not use this scene: it places the robot at the origin with its own
  table and cups; running one of its policies here needs those relative positions to match.

## Not done yet

- Container position is fixed for now; it will be spawned at random positions later. Since 2026-11-26 it
  stands on the board here, while the kitchen pour training (`CUP_CENTRE_*`) still uses the bottom table spot.
- Container is in the MuJoCo preview only; Isaac needs a container USD first.
- Container handle points at the robot axis (yaw computed in `container_pose()`).
