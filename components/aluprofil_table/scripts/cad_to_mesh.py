# HestiaSimulation (RoboTUM), author: Jeffrey

"""Prepare a downloaded CAD file (STEP/STP/STL/OBJ/...) for Isaac Lab's mesh converter.

Converts to OBJ, scales from CAD units (usually millimeters) to meters, optionally rotates Y-up
models to Z-up, and moves the origin to the floor center (min z = 0, centered in x/y).

STEP needs a CAD kernel (``cascadio``); pull it in temporarily with ``uv run --with``:

.. code-block:: bash

    uv run --with cascadio --with mapbox-earcut python ~/aluprofil_table/cad_to_mesh.py \
        ~/Downloads/table.step ~/aluprofil_table/meshes/downloaded_table.obj --unit mm
"""

import argparse
import os

import numpy as np
import trimesh

UNIT_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254}


def main():
    parser = argparse.ArgumentParser(description="Convert a CAD file to a meter-scaled, Z-up OBJ.")
    parser.add_argument("input", type=str, help="Input CAD/mesh file (.step, .stp, .stl, .obj, .glb, ...).")
    parser.add_argument("output", type=str, help="Output mesh file (.obj recommended).")
    parser.add_argument("--unit", choices=list(UNIT_SCALE), default="mm", help="Unit of the input file.")
    parser.add_argument("--y-up", action="store_true", help="Rotate a Y-up model to Z-up.")
    parser.add_argument("--no-recenter", action="store_true", help="Keep the original origin.")
    args = parser.parse_args()

    scene = trimesh.load(args.input, force="scene")
    mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
    print(f"[INFO] Loaded {args.input}: {len(scene.geometry)} bodies, {len(mesh.faces)} faces")

    mesh.apply_scale(UNIT_SCALE[args.unit])
    if args.y_up:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0)))
    if not args.no_recenter:
        lo, hi = mesh.bounds
        mesh.apply_translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))

    extents = mesh.extents
    print(f"[INFO] Size [m]: x={extents[0]:.3f} y={extents[1]:.3f} z={extents[2]:.3f}")
    if extents.max() > 10 or extents.max() < 0.05:
        print("[WARN] Size looks off for a table - check --unit.")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    mesh.export(args.output)
    print(f"[INFO] Wrote {args.output}")


if __name__ == "__main__":
    main()
