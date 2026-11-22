# Aluprofil table

Lab table built from 80x80 aluminium profiles (slot 10), measured on the real table 2026-11-22.

| | |
|---|---|
| Frame | 1800 x 960 mm, 3 front legs (500 mm), 3 back uprights (1320 mm), 85 mm levelling feet |
| Top | 1800 x 840 mm, 7 planks of 120 x 15 mm, slots 10 mm every 30 mm (M6 T-nuts), flush with the front |
| Heights | table top 680 mm, highest point (back wall) 1535 mm |
| Extras | back wall 1800 x 420 x 10 (R20), angle brackets, lower rails; type `standard` also has two 40x40 diagonal braces |
| Confirmed details | middle back upright has a levelling foot like the corners; the front rail is one 1640 mm piece over the middle front leg; lower rails on both sides (left and right); angle brackets on both sides of the middle back upright; bracket size 80 x 80 (accepted, not measured) |

| File | Content |
|---|---|
| `alu_table_jeffrey_v1_nov22.stl` | type **standard** (with diagonal braces), **mm**, Z-up, origin on the floor at the table centre, front edge at +y |
| `alu_table_no_brace_jeffrey_v1_nov22.stl` | type **no_brace**: the same table without the diagonal braces |
| `drawings/` | dimensioned drawing, settings sheet (PDF + PNG) |
| `scripts/` | generator: change the table and regenerate everything (see `scripts/README.md`) |

Simulation (MuJoCo / Isaac Lab) uses metres: `scripts/make_table.py` writes `meshes/alu_table.obj`
(m), which Isaac Lab converts to `usd/alu_table.usd` (both generated, not committed).
Scenes that use this table: `scenes/u_table_layout` (type `no_brace`).
