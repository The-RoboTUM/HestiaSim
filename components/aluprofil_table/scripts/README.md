# Aluprofil table: generator scripts

The table (80x80 aluminium profiles, slotted top, back wall, braces, brackets) is generated
from these scripts. Change a value, rerun, and the STL files and drawings in the folder above are
regenerated. All lengths on the command line are in **meters**; STL files are written in **mm**.

Run everything from this `scripts/` folder; results go one level up (`../`, `../drawings/`).
`uv` pulls in the dependencies for each run:

```bash
B="uv run --no-project --with trimesh --with shapely --with mapbox-earcut --with matplotlib"

$B python export_stl.py --suffix _v<N>          # ../alu_table_*.stl (mm)
$B python make_drawing.py --suffix _v<N>        # ../drawings/alu_table_drawing_*.pdf/.png
$B python make_settings_sheet.py --suffix _v<N> # ../drawings/alu_table_settings_*.pdf/.png

# laptop preview without Isaac (MuJoCo viewer); --drop-test --steps 300 = headless check with test cubes
$B --with mujoco --with scipy --with networkx python preview_mujoco.py
```

**Table types** (`--type`, every script): `standard` = the measured lab table with the two diagonal braces;
`no_brace` = the same table without the braces (and without their plank notches). Output files carry the
type in their name, e.g. `alu_table_no_brace_v2.stl`, `meshes/alu_table_no_brace.obj`.

Every script takes the same table options as `make_table.py` (see `python make_table.py --help`),
e.g. `--leg 0.60 --back-leg 1.42` for a top 100 mm higher. Defaults = the real lab table, measured
2026-11-22: 1800 x 960 frame, top at 680 mm, highest point 1535 mm.

File names follow the repo pattern `<part>_v<N>` (see the repo README), e.g. `--suffix _v2`.

| Script | Does |
|---|---|
| `make_table.py` | builds all table parts; writes `../meshes/alu_table.obj` (meters, for MuJoCo / Isaac) |
| `export_stl.py` | STL in mm for the hardware team (`--parts` also writes every part separately) |
| `make_drawing.py` | dimensioned A3 drawing with an empty "Real" column for checking measurements |
| `make_settings_sheet.py` | sheet of the configurable settings (`--detailed` adds calculated values) |
| `preview_mujoco.py` | MuJoCo preview, no NVIDIA GPU needed (`--drop-test`: test cubes on the top) |
| `spawn_table.py` | Isaac Lab: spawn the table USD (needs Isaac Lab; `--drop-test` as above) |
| `cad_to_mesh.py` | turn a downloaded CAD/mesh file (STEP/STL/OBJ, mm) into a meter-scaled OBJ |
| `paths.py` | where results are written |

## Isaac Lab

Isaac Lab spawns USD. Convert the meter-based OBJ once per design change (from your `IsaacLab` folder):

```bash
uv run python scripts/tools/convert_mesh.py <component>/meshes/alu_table.obj \
    <component>/usd/alu_table.usd --collision-approximation triangleMesh
# the no_brace type (used by scenes/u_table_layout): make_table.py --type no_brace, then
uv run python scripts/tools/convert_mesh.py <component>/meshes/alu_table_no_brace.obj \
    <component>/usd/alu_table_no_brace.usd --collision-approximation triangleMesh
uv run python <component>/scripts/spawn_table.py --viz kit   # add --drop-test: cubes should rest at 0.72 m
```

Don't use the mm STL for Isaac without scaling it by 0.001.
