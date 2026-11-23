"""Round adapter plate between the table top and a Franka Panda arm (meters, Z-up, origin = plate
centre on its bottom face) -> meshes/panda_adapter.obj + drawings/panda_adapter.png/.pdf.

Measured on the real plate (2026-11-22): radius 175 mm (Ø350), 24 counterbored holes on a 165 mm
radius (Ø330 bolt circle), counterbore Ø15 x 5.8 deep / through hole Ø8.4 = standard M8 clearance, ISO 273 fine (caliper read
8.32, then 8.0; a true Ø8.0 is too tight for an M8 screw) -> for M8
screws, plus one more such hole in the centre. The ring is turned so that an edge hole lies exactly
on the robot's centre line, at the front and at the back (confirmed on the real plate).

Robot pattern from the Franka Panda datasheet ("Arm footprint", May 2018): 4 x Ø9 holes in the
arm's base (a pair 160 mm apart on the round column side = robot FRONT, a pair 80 mm apart at the
narrow connector end = robot BACK, rows 127 mm apart) and 2 x Ø6 H7 pins 134 mm apart, 77 mm from
the 160 row towards the back. Rows 127, pins Ø6 / 134 / 77 confirmed on the real plate (2026-11-22). The plate has Ø8.4 through holes at the bolt positions (through holes
confirmed on the real plate; caliper read 8.0, set to the standard M8 clearance) and Ø6 pin holes.

Robot pattern position: the 160 mm bolt row is 20 mm from the middle hole's centre, towards the
connector end (measured). From the Panda base mesh (MuJoCo Menagerie link0: round end = circle r 72 mm
around joint 1) and the datasheet (round end 83 mm beyond the 160 row), joint 1 (the robot's vertical
axis) is 11.5 mm from the 160 row towards the front, so it sits 8.5 mm off the plate centre, towards
the connector end: (0, +0.0085) in plate coordinates.

Correction (2026-11-22): the previous version assumed the robot was centred on the plate and set the
160 row to 11.5 mm from the centre; the real plate measures 20 mm, so the robot is 8.5 mm off-centre.

Correction (2026-11-22): earlier versions called the 160 pair the robot's "rear" and the 80 pair
its "front", and put the 160 row 20 mm (guess) and then 0 mm from the centre. The 80 pair is at
the connector end (robot back); the 0 mm reading did not account for the joint-1 position.

Plate thickness 10 mm (measured). The 24 + 1 counterbored holes are plain through holes: the M8
screws go through the plate into T-nuts in the table top.

Correction (2026-11-22): the 4 robot bolt holes were first modelled as M8 threads (Ø6.8 tap drill).
They are through holes (checked on the real plate), Ø8.4 like the other M8 holes.

.. code-block:: bash

    uv run --no-project --with trimesh --with shapely --with mapbox-earcut --with matplotlib \\
        python make_adapter.py
"""

import argparse
import os

import numpy as np
import trimesh  # noqa: E402
from shapely.geometry import Point  # noqa: E402

from paths import DRAWINGS, MESHES, STL_OUT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
JOINT1_TO_ROW160 = 0.0115  # m: joint 1 lies this far from the 160 row towards the robot front
ROBOT_HOLE = 0.0084  # robot bolt through hole: M8 clearance (caliper read 8.0)


def circle(x: float, y: float, d: float):
    return Point(x, y).buffer(d / 2, resolution=24)


def hole_layout(a) -> dict[str, list[tuple[float, float, float]]]:
    """Hole groups as (x, y, diameter); +y = robot BACK (connector end), -y = robot front,
    robot centre line = x = 0, joint 1 at (0, a.row160_offset - JOINT1_TO_ROW160)."""
    ring = [
        (a.bolt_radius * np.cos(t), a.bolt_radius * np.sin(t), a.through)
        for t in np.radians(np.arange(a.n_holes) * 360.0 / a.n_holes + 90.0)
    ]
    r = a.row160_offset  # y of the robot's 160 mm bolt row (front pair)
    robot = [(-0.080, r, ROBOT_HOLE), (0.080, r, ROBOT_HOLE),
             (-0.040, r + 0.127, ROBOT_HOLE), (0.040, r + 0.127, ROBOT_HOLE)]
    pins = [(-0.067, r + 0.077, 0.006), (0.067, r + 0.077, 0.006)]
    return {"ring": ring, "centre": [(0.0, 0.0, a.through)], "robot": robot, "pins": pins}


def build_plate(a) -> trimesh.Trimesh:
    holes = hole_layout(a)
    counterbored = holes["ring"] + holes["centre"]
    plain = holes["robot"] + holes["pins"]
    disk = Point(0, 0).buffer(a.radius, resolution=96)
    # lower layer: through holes; upper layer: counterbores (Ø15) + the same plain holes
    lower = disk
    for x, y, d in counterbored + plain:
        lower = lower.difference(circle(x, y, d))
    upper = disk
    for x, y, _ in counterbored:
        upper = upper.difference(circle(x, y, a.counterbore))
    for x, y, d in plain:
        upper = upper.difference(circle(x, y, d))
    t_low = a.thickness - a.counterbore_depth
    low = trimesh.creation.extrude_polygon(lower, t_low)
    up = trimesh.creation.extrude_polygon(upper, a.counterbore_depth)
    up.apply_translation((0, 0, t_low))
    return trimesh.util.concatenate([low, up])


def drawing(a, out_base: str):
    import matplotlib  # only needed for the drawing (scenes import build_plate() without it)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    holes = hole_layout(a)
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.add_patch(plt.Circle((0, 0), a.radius * 1000, fc="#dfe3e8", ec="black", lw=0.8))
    ax.add_patch(plt.Circle((0, 0), a.bolt_radius * 1000, fill=False, ec="#888", lw=0.5, ls="-."))
    style = {"ring": ("#ffffff", f"Ø{a.through * 1000:.1f} / cbore Ø15 (M8 screw to table)"),
             "centre": ("#ffffff", None),
             "robot": ("#f2c48d", "Ø8.4 through (robot bolts M8x25)"),
             "pins": ("#9ec3e6", "Ø6 H7 (robot locating pins)")}
    for group, pts in holes.items():
        fc, label = style[group]
        for i, (x, y, d) in enumerate(pts):
            if group in ("ring", "centre"):
                ax.add_patch(plt.Circle((x * 1000, y * 1000), a.counterbore * 500, fill=False, ec="black", lw=0.4))
            ax.add_patch(plt.Circle((x * 1000, y * 1000), d * 500, fc=fc, ec="black", lw=0.6,
                                    label=label if i == 0 else None))
    r = a.row160_offset * 1000
    txt = dict(fontsize=8, ha="center")
    ax.annotate("", (-80, r - 25), (80, r - 25), arrowprops=dict(arrowstyle="<->", lw=0.6))
    ax.text(0, r - 40, "160", **txt)
    ax.annotate("", (-40, r + 127 - 14), (40, r + 127 - 14), arrowprops=dict(arrowstyle="<->", lw=0.6))
    ax.text(0, r + 127 - 28, "80", **txt)
    ax.annotate("", (-67, r + 77 - 14), (67, r + 77 - 14), arrowprops=dict(arrowstyle="<->", lw=0.6))
    ax.text(0, r + 77 - 28, "134", **txt)
    ax.annotate("", (110, r), (110, r + 127), arrowprops=dict(arrowstyle="<->", lw=0.6))
    ax.text(118, r + 63, "127", fontsize=8, rotation=90, va="center")
    ax.annotate("", (95, r), (95, r + 77), arrowprops=dict(arrowstyle="<->", lw=0.6))
    ax.text(82, r + 38, "77", fontsize=8, rotation=90, va="center")
    if abs(r) > 0.5:
        ax.annotate("", (-150, 0), (-150, r), arrowprops=dict(arrowstyle="<->", lw=0.6))
        ax.text(-160, r / 2, f"{r:.1f}", fontsize=8, rotation=90, va="center")
    else:
        ax.plot([-95, 95], [0, 0], color="#888", lw=0.5, ls="--")
    ax.text(0, -a.radius * 1000 - 42, f"Ø{2 * a.radius * 1000:.0f} plate, {a.n_holes} holes on Ø{2 * a.bolt_radius * 1000:.0f}"
            f" bolt circle, thickness {a.thickness * 1000:.0f} mm, cbore Ø{a.counterbore * 1000:.0f} x {a.counterbore_depth * 1000:.1f} deep", **txt)
    ax.text(0, a.radius * 1000 + 15, "robot BACK (connector end)", fontsize=8, ha="center", color="#666")
    ax.text(0, -a.radius * 1000 - 12, "robot front", fontsize=8, ha="center", color="#666", va="top")
    j1 = r - JOINT1_TO_ROW160 * 1000
    ax.plot([-10, 10], [j1, j1], color="#b00020", lw=0.8)
    ax.plot([0, 0], [j1 - 10, j1 + 10], color="#b00020", lw=0.8)
    ax.text(12, j1 - 4, "joint 1 axis", fontsize=7, color="#b00020", va="top")
    ax.text(a.radius * 1000 + 15, a.radius * 1000 - 90,
            "Robot pattern: Franka Panda datasheet,\nArm footprint (May 2018).\nAll M8 holes Ø8.4 = ISO 273 fine clearance\n(caliper read 8.0).",
            fontsize=7, color="#b00020", va="top")
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.set_xlim(-a.radius * 1000 - 40, a.radius * 1000 + 230)
    ax.set_ylim(-a.radius * 1000 - 55, a.radius * 1000 + 35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("PANDA ADAPTER PLATE - top view, mm", fontsize=10, loc="left")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Panda adapter plate (all lengths in meters).")
    p.add_argument("--radius", type=float, default=0.175, help="Plate radius (measured 175 mm).")
    p.add_argument("--bolt-radius", type=float, default=0.165, help="Radius of the hole circle (measured 165 mm).")
    p.add_argument("--n-holes", type=int, default=24, help="Holes on the circle (counted on the photo: 24).")
    p.add_argument("--through", type=float, default=0.0084, help="Through hole: M8 clearance 8.4 (caliper read 8.0).")
    p.add_argument("--counterbore", type=float, default=0.015, help="Counterbore diameter (measured 15 mm).")
    p.add_argument("--counterbore-depth", type=float, default=0.0058, help="Counterbore depth (measured 5.8 mm).")
    p.add_argument("--thickness", type=float, default=0.010, help="Plate thickness (measured 10 mm).")
    p.add_argument("--row160-offset", type=float, default=0.020,
                   help="y of the robot's 160 mm bolt row from the plate centre [m] (measured 20 mm).")
    p.add_argument("--suffix", type=str, default="", help="Appended to drawing/STL names, e.g. _v2.")
    p.add_argument("--stl", action="store_true", help="Also write the plate as STL in mm (hand-off file).")
    p.add_argument("--output", type=str, default=os.path.join(MESHES, "panda_adapter.obj"))
    return p


def main():
    a = build_parser().parse_args()
    plate = build_plate(a)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    plate.export(a.output)
    out = os.path.join(DRAWINGS, f"panda_adapter_drawing{a.suffix}" if a.suffix else "panda_adapter")
    os.makedirs(DRAWINGS, exist_ok=True)
    drawing(a, out)
    print(f"[INFO] plate {plate.extents.round(4)} m, watertight={plate.is_watertight}, volume={plate.volume * 1e6:.0f} cm3")
    print(f"[INFO] Wrote {a.output} and {out}.png/.pdf")
    if a.stl:
        stl = plate.copy()
        stl.apply_scale(1000.0)
        path = os.path.join(STL_OUT, f"panda_adapter{a.suffix or '_mm'}.stl")
        os.makedirs(STL_OUT, exist_ok=True)
        stl.export(path)
        print(f"[INFO] Wrote {path} (mm)")


if __name__ == "__main__":
    main()
