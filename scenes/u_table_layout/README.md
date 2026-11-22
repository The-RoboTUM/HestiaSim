# U table layout

Three aluprofil tables of type **`no_brace`** (no diagonal braces) in a U, with the mixer on the left table
and the Panda adapter plate with a Franka Panda on the left table too. This scene only **references** the
components (`components/aluprofil_table`, `components/mixer`, `components/panda_adapter`, `components/franka_panda`); it never copies their geometry, so a
new table or mixer version shows up here automatically.

```
Left / right = as seen standing in the opening of the U, looking at the bottom table.

       ┌──────────────────┐
 ┌────┐│   TableBottom    │┌────┐   back walls outside, fronts facing the inside;
 │    │└──────────────────┘│    │   side tables outside the ends of the bottom table,
 │ R  │      inside    ◄─M │ L  │   their back ends halfway along the bottom table's depth
 │    │                    │    │   M = mixer in the middle of the left table, perpendicular
 └────┘                    └────┘       to it, front (bowl side) facing the inside of the U
                                      A = Panda adapter on the left table at its end next to the
                                          bottom table, robot facing 45° "south-east" (towards the
                                          inside and the opening of the U)
             ▲ you stand here
```

Origin = centre of the bottom table on the floor, metres, Z-up. Compass as in the viewer's start view:
north = towards the bottom table (-y), south = towards the opening (+y), east = -x, west = +x.


| Object | x [m] | y [m] | yaw | z |
|---|---|---|---|---|
| TableBottom | 0 | 0 | 0° | floor |
| TableLeft | +1.38 | +0.90 | +90° | floor |
| TableRight | -1.38 | +0.90 | -90° | floor |
| Mixer (mum8, Mana v1 = her newest, confirmed) | +1.25 | +0.90 | 180° (front facing the inside of the U) | on the left table top (0.68) |
| Panda adapter (Ø350 x 10) | +1.175 | +0.250 | -135° (robot faces +135°: "south-east") | on the left table top; edge 75 mm from its end next to the bottom table, 100 mm from its front edge |
| Franka Panda (`components/franka_panda` v1) | +1.181 | +0.244 | robot front +135° ("south-east") | 0.69 (on the adapter), home pose |

Overall 3720 x 2280 mm, inside opening 1800 mm. Change positions in `u_layout.py`
(`TABLE_TYPE`, `placements()`, `MIXER_ON`; the mixer stands perpendicular to its table, front to the inside) or with the options `--gap`, `--side-offset`, `--no-mixer`, `--no-adapter`, `--no-robot`; adapter position via `ADAPTER_TABLE`, `ADAPTER_END`, `ADAPTER_*` and `ROBOT_FACING_DEG`.

| File | Does |
|---|---|
| `u_layout.py` | positions of the tables and the mixer (pure Python, used by both scripts) |
| `preview_U_table_layout.py` | MuJoCo preview on a laptop (mixer shown for placement only, no collision) |
| `spawn_U_table_layout.py` | Isaac Lab: table USD x3 + mixer USD + adapter USD + Isaac Lab's Franka Panda |

```bash
# laptop (from this folder)
UV="uv run --no-project --with mujoco --with trimesh --with shapely --with mapbox-earcut --with scipy --with networkx"
$UV python preview_U_table_layout.py                          # viewer
$UV python preview_U_table_layout.py --drop-test --steps 300  # check: 9 test cubes rest at 0.72 m

# Isaac Lab workstation (from your IsaacLab folder). Needs the no_brace table USD once:
# see components/aluprofil_table/scripts/README.md (convert meshes/alu_table_no_brace.obj), and the same for
# components/panda_adapter/meshes/panda_adapter.obj -> usd/panda_adapter.usd
uv run python <HestiaSimulation>/scenes/u_table_layout/spawn_U_table_layout.py --visualizer kit
```

In Isaac the mixer's bowl and lid are rigid bodies (they have mass in the mixer's USD), the machine
body is a static collider.

## Robot model

The Panda comes from **`components/franka_panda`** (`franka_cfg.py`, settings v1 = the robot the pouring
training `training/hestia_beans_pour` uses: Isaac Lab's Panda, gravity off, arm stiffness 1000/750/300,
damping 20/4/2, the pouring start pose; not final yet). When the training changes the robot, only that
component changes; this scene follows.

- **Start pose in this scene:** the Panda's upright **home** pose (0, 0, 0, -1.571, 0, 1.571, -0.785; `ROBOT_SCENE_POSE`
  in `u_layout.py`), not the training's start pose (folded low near the table, only useful with its cups).
  Servo values and gravity off still come from the component.
- **Isaac Lab:** `franka_cfg.isaac_cfg()` (Isaac Lab's `FRANKA_PANDA_CFG` + those settings), held at the home pose.
- **MuJoCo preview:** the same settings (`franka_cfg.mujoco_settings()`) on MuJoCo Menagerie's
  `franka_emika_panda/panda.xml` (same Franka description: same links and joints). Same robot, pose and servo
  values; the physics engines differ, so test trained policies in Isaac Lab. Menagerie is not part of this repo.
  Default path: `~/Projects/RoboTUM/franka-sandbox/third_party/mujoco_menagerie/...`; set `HESTIA_PANDA_MJCF` or
  pass `--panda-xml`. Get it with
  `git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git`
  then `git sparse-checkout set franka_emika_panda`.

Mounting pose (both): on the adapter, `robot_pose()` in `u_layout.py`: x +1.181, y +0.244, z 0.69, facing +135°.
The training itself places the robot at the origin with its own table and cups; running a trained policy in
this scene needs those relative positions to match as well.
