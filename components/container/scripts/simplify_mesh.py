"""Make a simplified copy of a heavy mesh for simulation, and check how accurate it is.

Used for container01_v3.stl (50k triangles) from container01_v2.stl (1.6M triangles). Works for any STL.
Keeps the units and the origin: only the number of triangles changes.

.. code-block:: bash

    cd components/container/scripts
    uv run --no-project --with trimesh --with numpy --with scipy --with rtree --with fast-simplification \
        python simplify_mesh.py ../container01_v2.stl ../container01_v3.stl --faces 50000

Prints the accuracy check: size (bounding box), volume, and how far the simplified mesh's vertices lie from
the original surface (exact distance, measured on 20k vertices). Accept the result if the size is unchanged
and the distance is far below what matters in simulation (for the container: 0.024 mm max).
"""

import argparse

import numpy as np
import trimesh


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("src", help="original mesh (e.g. ../container01_v2.stl)")
    p.add_argument("dst", help="simplified copy to write (e.g. ../container01_v3.stl)")
    p.add_argument("--faces", type=int, default=50_000, help="target number of triangles (default 50000)")
    a = p.parse_args()

    src = trimesh.load(a.src, force="mesh")
    m = src.simplify_quadric_decimation(face_count=a.faces)  # quadric edge collapse (fast-simplification)
    m.merge_vertices()
    m.export(a.dst)

    # accuracy check
    rng = np.random.default_rng(0)
    idx = rng.choice(len(m.vertices), min(20_000, len(m.vertices)), replace=False)
    _, dist, _ = trimesh.proximity.closest_point(src, m.vertices[idx])
    _, counts = np.unique(m.edges_sorted, axis=0, return_counts=True)
    print(f"triangles      {len(src.faces)} -> {len(m.faces)}")
    print(f"bounds         {src.bounds.round(2).tolist()} -> {m.bounds.round(2).tolist()}")
    print(f"volume         {m.volume / src.volume * 100:.3f} % of the original")
    print(f"vertex -> original surface: mean {dist.mean():.4f}, max {dist.max():.4f} (mesh units)")
    print(f"open edges (holes) {np.sum(counts == 1)}, edges shared by >2 triangles {np.sum(counts > 2)}")
    print(f"wrote {a.dst}")


if __name__ == "__main__":
    main()
