"""Export the table (and the Panda adapter plate) as STL in MILLIMETERS for CAD / hardware people.

STL has no units; CAD tools assume mm, so the meter-based model is scaled x1000 here.
Outputs (exports/ in the dev folder, the component folder when run from <component>/scripts/):
    alu_table[_<type>]<suffix>.stl whole table, one mesh (--type no_brace -> alu_table_no_brace...)
    panda_adapter<suffix>.stl      adapter plate (from make_adapter.py defaults, if that script is present)
    parts/<part>.stl               every table part separately (only with --parts)

The HestiaSim repo names files <part>_v<N> (see the repo README), e.g.
    python export_stl.py --suffix _v2

.. code-block:: bash

    uv run --no-project --with trimesh --with shapely --with mapbox-earcut --with matplotlib \\
        python export_stl.py [--suffix _mm] [--parts]      # plus any make_table.py flags
"""

import os
import shutil
import sys

import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:  # the adapter plate is optional (not every copy of the scripts has make_adapter.py)
    import make_adapter  # noqa: E402
except ImportError:
    make_adapter = None
from make_table import build_from_args, build_parser, type_tag  # noqa: E402
from paths import STL_OUT  # noqa: E402


def main():
    parser = build_parser()
    parser.add_argument("--suffix", type=str, default="_mm", help="Appended to the file names (before .stl).")
    parser.add_argument("--out-dir", type=str, default=STL_OUT, help="Where to write the STL files.")
    parser.add_argument("--parts", action="store_true", help="Also write every part as its own STL in parts/.")
    args = parser.parse_args()
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    parts = build_from_args(args)
    if args.parts:
        parts_dir = os.path.join(out, "parts")
        shutil.rmtree(parts_dir, ignore_errors=True)  # no stale parts from an older design
        os.makedirs(parts_dir)
        for name, mesh in parts.items():
            m = mesh.copy()
            m.apply_scale(1000.0)
            m.export(os.path.join(parts_dir, f"{name}.stl"))
    table = trimesh.util.concatenate(list(parts.values()))
    table.apply_scale(1000.0)
    table_path = os.path.join(out, f"alu_table{type_tag(args)}{args.suffix}.stl")
    table.export(table_path)
    written = [table_path]
    print(f"[INFO] table: {len(parts)} parts, bounds [mm] {table.bounds.round(1).tolist()}")
    if make_adapter is not None:
        plate = make_adapter.build_plate(make_adapter.build_parser().parse_args([]))
        plate.apply_scale(1000.0)
        plate_path = os.path.join(out, f"panda_adapter{args.suffix}.stl")
        plate.export(plate_path)
        written.append(plate_path)
        print(f"[INFO] adapter: extents [mm] {plate.extents.round(1).tolist()}")
    print(f"[INFO] Wrote {' and '.join(written)}" + (f" + {len(parts)} files in parts/" if args.parts else ""))


if __name__ == "__main__":
    main()
