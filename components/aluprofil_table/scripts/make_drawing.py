# HestiaSimulation (RoboTUM), author: Jeffrey

"""Dimensioned A3 drawing of the table (front, side, top view + measurement table), in mm.

Takes the same options as make_table.py, so the drawing always matches the model. The table on the
right has an empty "Real" column to write the measured values into; feed them back as flags.

.. code-block:: bash

    uv run --with mapbox-earcut --with matplotlib python ~/aluprofil_table/make_drawing.py
    # -> ~/aluprofil_table/drawings/alu_table_drawing.pdf / .png
"""

import datetime
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402
from shapely.geometry import MultiPoint, box  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_table import build_from_args, build_parser, type_tag  # noqa: E402
from paths import DRAWINGS  # noqa: E402

COLORS = {
    "foot": "#555555",
    "leg": "#b8bcc4",
    "rail": "#c9ccd2",
    "crossbar": "#c9ccd2",
    "plate_plank": "#9ec3e6",
    "back_plate": "#f2c48d",
    "brace": "#8a8f99",
    "bracket": "#3a3d42",
}
TXT = dict(fontsize=7, ha="center", va="center")


def color_for(name: str) -> str:
    for key, c in COLORS.items():
        if key in name:
            return c
    return "#cccccc"


def draw_view(ax, parts, i: int, j: int, depth: int, flip_depth: bool, radius: float):
    """Project part bounding boxes onto axes (i, j), far parts first."""
    items = sorted(parts.items(), key=lambda kv: kv[1].bounds[1][depth] * (-1 if flip_depth else 1))
    for name, mesh in items:
        b = mesh.bounds * 1000.0
        shape = box(b[0][i], b[0][j], b[1][i], b[1][j])
        if (name.startswith("brace_") and "joint" not in name) or name.startswith("bracket_"):  # diagonal: real outline, not its bbox
            shape = MultiPoint((mesh.vertices[:, [i, j]] * 1000.0).tolist()).convex_hull
        if name == "back_plate" and i == 0 and j == 2 and radius > 0:
            shape = shape.buffer(-radius).buffer(radius, resolution=8)
        ax.add_patch(
            MplPolygon(list(shape.exterior.coords), closed=True, fc=color_for(name), ec="black", lw=0.5, alpha=0.75)
        )


def dim_h(ax, x1, x2, y, ref1, ref2, text, small=False):
    """Horizontal dimension between x1 and x2 drawn at height y, extension lines from ref1/ref2."""
    for x, r in ((x1, ref1), (x2, ref2)):
        ax.plot([x, x], [r, y], color="#444", lw=0.4)
    ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle="<->", lw=0.6, shrinkA=0, shrinkB=0))
    ax.text((x1 + x2) / 2, y + 28, text, **TXT, color="#b00020" if small else "black")


def dim_v(ax, y1, y2, x, ref1, ref2, text, small=False):
    """Vertical dimension between y1 and y2 drawn at x, extension lines from ref1/ref2."""
    for y, r in ((y1, ref1), (y2, ref2)):
        ax.plot([r, x], [y, y], color="#444", lw=0.4)
    ax.annotate("", xy=(x, y2), xytext=(x, y1), arrowprops=dict(arrowstyle="<->", lw=0.6, shrinkA=0, shrinkB=0))
    ax.text(x - 30, (y1 + y2) / 2, text, rotation=90, **TXT, color="#b00020" if small else "black")


def main():
    parser = build_parser()
    parser.description = "Dimensioned drawing of the aluminium-profile table."
    parser.set_defaults(output=None)
    parser.add_argument("--suffix", type=str, default="", help="Appended to the file name, e.g. _v2.")
    args = parser.parse_args()
    parts = build_from_args(args)

    m = 1000.0
    L, W, S = args.length * m, args.width * m, args.profile * m
    F, LG, P = args.foot * m, args.leg * m, args.plate * m
    BL, BH, BT = args.back_leg * m, args.back_plate_height * m, args.back_plate_thickness * m
    BO, R = args.back_plate_over * m, args.back_plate_radius * m
    rail_top = F + LG + S
    top = rail_top + P
    up_top, cross_top = F + BL, F + BL + S
    bp_top = cross_top + BO
    bp_bot = bp_top - BH
    tall = "back_crossbar" in parts
    zmax = bp_top if "back_plate" in parts else (cross_top if tall else top)

    planks = sorted((v.bounds * m for k, v in parts.items() if k.startswith("plate_plank")), key=lambda b: b[0][1])
    py0 = min(b[0][1] for b in planks) if planks else -W / 2
    py1 = max(b[1][1] for b in planks) if planks else W / 2

    fig = plt.figure(figsize=(16.54, 11.69))  # A3 landscape
    gs = fig.add_gridspec(
        2, 3, width_ratios=[L + 1100, W + 700, 2600], height_ratios=[zmax + 500, W + 700], wspace=0.02, hspace=0.04
    )
    ax_f, ax_s, ax_t = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])
    ax_tab = fig.add_subplot(gs[:, 2])
    for ax in (ax_f, ax_s, ax_t, ax_tab):
        ax.set_axis_off()

    # ---- front view (x, z), seen from the front (+y) ----
    draw_view(ax_f, parts, 0, 2, 1, False, R)
    ax_f.plot([-L / 2 - 650, L / 2 + 450], [0, 0], color="black", lw=1.0)
    dim_h(ax_f, -L / 2, L / 2, -150, 0, 0, f"{L:.0f}")
    x1, x2 = L / 2 + 150, L / 2 + 330
    dim_v(ax_f, 0, F, x1, L / 2, L / 2, f"{F:.0f}", small=True)
    dim_v(ax_f, F, F + LG, x1, L / 2, L / 2, f"{LG:.0f}")
    dim_v(ax_f, F + LG, rail_top, x1, L / 2, L / 2, f"{S:.0f}", small=True)
    dim_v(ax_f, rail_top, top, x1 + 90, L / 2, L / 2, f"{P:.0f}", small=True)
    dim_v(ax_f, 0, top, x2, L / 2, L / 2, f"{top:.0f}")
    if tall:
        xl1, xl2, xl3 = -L / 2 - 150, -L / 2 - 330, -L / 2 - 510
        dim_v(ax_f, F, up_top, xl1, -L / 2, -L / 2, f"{BL:.0f}")
        dim_v(ax_f, up_top, cross_top, xl1, -L / 2, -L / 2, f"{S:.0f}", small=True)
        if "back_plate" in parts:
            dim_v(ax_f, cross_top, bp_top, xl1 - 90, -L / 2, -L / 2, f"{BO:.0f}", small=True)
            dim_v(ax_f, bp_bot, bp_top, xl2, -L / 2, -L / 2, f"{BH:.0f}")
        dim_v(ax_f, 0, zmax, xl3, -L / 2, -L / 2, f"{zmax:.0f}")
        if R > 0:
            ax_f.text(L / 2 - 60, bp_top - 60, f"R{R:.0f}", fontsize=7, ha="right", va="top", color="#b00020")
    ax_f.set_title("FRONT VIEW", fontsize=10, loc="left")
    ax_f.set_xlim(-L / 2 - 700, L / 2 + 450)
    ax_f.set_ylim(-260, zmax + 240)
    ax_f.set_aspect("equal")

    # ---- side view (y, z), seen from the right (+x) ----
    draw_view(ax_s, parts, 1, 2, 0, False, R)
    ax_s.plot([-W / 2 - 300, W / 2 + 350], [0, 0], color="black", lw=1.0)
    dim_h(ax_s, -W / 2, W / 2, -150, 0, 0, f"{W:.0f}")
    dim_h(ax_s, py0, py1, top + 110, top, top, f"{py1 - py0:.0f}")
    if py0 > -W / 2 + 1:
        dim_h(ax_s, -W / 2, py0, top + 110, rail_top, top, f"{py0 + W / 2:.0f}", small=True)
    dim_h(ax_s, W / 2 - S, W / 2, F + LG - 120, F + LG, F + LG, f"{S:.0f}", small=True)
    if "back_plate" in parts:
        b = parts["back_plate"].bounds * m
        dim_h(ax_s, b[0][1], b[1][1], bp_top + 90, bp_top, bp_top, f"{BT:.0f}", small=True)
    ax_s.set_title("SIDE VIEW (from right)", fontsize=10, loc="left")
    ax_s.set_xlim(-W / 2 - 300, W / 2 + 400)
    ax_s.set_ylim(-260, zmax + 240)
    ax_s.set_aspect("equal")

    # ---- top view (x, y), seen from above ----
    draw_view(ax_t, parts, 0, 1, 2, False, R)
    dim_h(ax_t, -L / 2, L / 2, -W / 2 - 150, -W / 2, -W / 2, f"{L:.0f}")
    dim_v(ax_t, -W / 2, W / 2, -L / 2 - 150, -L / 2, -L / 2, f"{W:.0f}")
    for b in planks:
        dim_v(ax_t, b[0][1], b[1][1], L / 2 + 150, L / 2, L / 2, f"{b[1][1] - b[0][1]:.0f}", small=True)
    ax_t.text(0, -W / 2 - 300, "front (+y) is at the top of this view", fontsize=7, ha="center", color="#666")
    ax_t.set_title("TOP VIEW", fontsize=10, loc="left")
    ax_t.set_xlim(-L / 2 - 700, L / 2 + 450)
    ax_t.set_ylim(-W / 2 - 380, W / 2 + 300)
    ax_t.set_aspect("equal")

    # ---- measurement table ----
    rows = [
        ("Frame length", "--length", L),
        ("Frame width", "--width", W),
        ("Profile size (square)", "--profile", S),
        ("Levelling foot height", "--foot", F),
        ("Front leg length", "--leg", LG),
        ("Table top thickness", "--plate", P),
        ("Table top width", "--plate-width", args.plate_width * m),
        ("Top plank width (full)", "--plank", args.plank * m),
        ("Back upright length", "--back-leg", BL),
        ("Back plate height", "--back-plate-height", BH),
        ("Back plate thickness", "--back-plate-thickness", BT),
        ("Back plate over crossbar", "--back-plate-over", BO),
        ("Back plate corner radius", "--back-plate-radius", R),
        ("= Table top height", "(result)", top),
        ("= Highest point", "(result)", zmax),
    ]
    cells = [[name, flag, f"{v:.0f}", ""] for name, flag, v in rows]
    tab = ax_tab.table(
        cellText=cells,
        colLabels=["Part", "Flag", "Model [mm]", "Real [mm]"],
        colWidths=[0.33, 0.37, 0.15, 0.15],
        loc="upper center",
        cellLoc="left",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(7.5)
    tab.scale(1.0, 2.0)
    for (r, c), cell in tab.get_celld().items():
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8e8e8")
        elif c == 1:
            cell.set_text_props(family="monospace")
        elif c == 3:
            cell.set_facecolor("#fffbe6")
    ax_tab.text(
        0.0,
        0.30,
        "All dimensions in mm (model values).\n"
        "Red = small dimensions.\n"
        "Write the measured values in the 'Real' column,\n"
        "then rebuild with the flags, e.g.\n"
        "  make_table.py --leg 0.52 --back-plate-thickness 0.02\n"
        "(flags take meters).",
        transform=ax_tab.transAxes,
        fontsize=8,
        va="top",
        family="monospace",
    )
    ax_tab.text(
        0.0,
        0.06,
        f"ALU-PROFILE TABLE ({args.type})  |  A3  |  not to scale when printed  |  {datetime.date.today().isoformat()}",
        transform=ax_tab.transAxes,
        fontsize=8,
        weight="bold",
    )

    if args.output is None:
        args.output = os.path.join(DRAWINGS, f"alu_table{type_tag(args)}_drawing{args.suffix}.pdf")
    out = os.path.abspath(args.output if args.output.endswith(".pdf") else os.path.splitext(args.output)[0] + ".pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    fig.savefig(os.path.splitext(out)[0] + ".png", dpi=110)
    print(f"[INFO] Wrote {out} and {os.path.splitext(out)[0]}.png")


if __name__ == "__main__":
    main()
