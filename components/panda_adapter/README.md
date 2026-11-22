# Panda adapter plate

Round plate between the table top and a Franka Panda arm, measured on the real plate 2026-11-22.

| | |
|---|---|
| Plate | Ø350 x 10 mm |
| To the table | 24 holes on a Ø330 circle + 1 in the centre, Ø8.4 through, counterbore Ø15 x 5.8 (M8 screws into T-nuts in the table top); one edge hole lies on the robot's centre line, front and back |
| Robot | 4 x Ø8.4 through (M8 x 25 screws): a pair 160 mm apart (robot front side) and a pair 80 mm apart (connector end), 127 mm between the rows; 2 x Ø6 pin holes, 134 mm apart, 77 mm from the 160 row (Franka datasheet, confirmed on the plate) |
| Robot position | the 160 row is 20 mm from the centre hole towards the connector end, so the robot's vertical axis (joint 1) is 8.5 mm off the plate centre |

| File | Content |
|---|---|
| `panda_adapter_jeffrey_v1_nov22.stl` | plate, **mm**, Z-up, origin = plate centre on its bottom face, +y = robot back (connector end) |
| `drawings/` | top view with all hole positions (PDF + PNG) |
| `scripts/make_adapter.py` | generator: `--suffix _<author>_v<N>_<date> --stl` writes the STL + drawing; also writes `meshes/panda_adapter.obj` (metres) for MuJoCo / Isaac |

Isaac Lab: convert `meshes/panda_adapter.obj` to `usd/panda_adapter.usd` once (like the table, see
`components/aluprofil_table/scripts/README.md`, static collider). Scenes that use it: `scenes/u_table_layout`.
