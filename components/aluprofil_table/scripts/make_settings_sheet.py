"""Settings sheets for the table, drawn on the real model geometry, values read from the model.

    drawings/alu_table_settings<suffix>.pdf/.png   simple sheet for others: only what can be configured
    drawings/config_proposal.pdf/.png     detailed version (--detailed): also what is calculated,
                                          the fixed parts and the checks (incl. a dishwasher box)

Takes the same flags as make_table.py, so the numbers on the sheet always match the model.

.. code-block:: bash

    uv run --no-project --with trimesh --with shapely --with mapbox-earcut --with matplotlib \\
        python make_settings_sheet.py [--detailed]
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_drawing import draw_view  # noqa: E402
from make_table import build_from_args, build_parser, type_tag  # noqa: E402
from paths import DRAWINGS  # noqa: E402

RED, BLUE, GREY = "#c0002a", "#1f5fbf", "#8a8f99"


def arrow_v(ax, x, z0, z1, txt, color, side="right", fs=9, bold=False):
    ax.annotate("", (x, z0), (x, z1), arrowprops=dict(arrowstyle="<->", color=color, lw=1.6 if bold else 1))
    ax.text(x + (18 if side == "right" else -18), (z0 + z1) / 2, txt, color=color, fontsize=fs,
            ha="left" if side == "right" else "right", va="center", fontweight="bold" if bold else "normal")


def arrow_h(ax, x0, x1, z, txt, color, fs=9, bold=False, dz=18):
    ax.annotate("", (x0, z), (x1, z), arrowprops=dict(arrowstyle="<->", color=color, lw=1.6 if bold else 1))
    ax.text((x0 + x1) / 2, z + dz, txt, color=color, fontsize=fs, ha="center", fontweight="bold" if bold else "normal")


def main():
    parser = build_parser()
    parser.add_argument("--detailed", action="store_true", help="Also show calculated values, fixed parts and checks.")
    parser.add_argument("--suffix", type=str, default="", help="Appended to the file name, e.g. _v2.")
    args = parser.parse_args()
    parts = build_from_args(args)
    B = {k: m.bounds * 1000 for k, m in parts.items()}
    top = B["plate_plank_0"][1][2]
    under = B["top_rail_x_p"][0][2]
    zmax = max(b[1][2] for b in B.values())
    L, D, T = args.length * 1000, args.width * 1000, args.plate_width * 1000
    H, C, BW = top, under, zmax - top
    detailed = args.detailed

    fig = plt.figure(figsize=(16.5, 10.5))
    axf = fig.add_axes([0.06, 0.06, 0.47, 0.86])
    axs = fig.add_axes([0.53, 0.06, 0.20, 0.86])
    for ax in (axf, axs):
        ax.set_aspect("equal")
        ax.axis("off")

    # ---- front view ----
    draw_view(axf, parts, 0, 2, 1, False, args.back_plate_radius * 1000)
    axf.plot([-L / 2 - 200, L / 2 + 250], [0, 0], color="k", lw=1)
    arrow_v(axf, -L / 2 - 100, 0, H, f"H  TABLE TOP HEIGHT\n   {H:.0f}", RED, side="left", bold=True, fs=10)
    arrow_v(axf, -L / 2 - 100, H, zmax, f"B  back wall above\n   the table top: {BW:.0f}", RED, side="left")
    arrow_v(axf, -L / 2 + 200, 0, C, f"C  free height under\n   the top: {C:.0f}\n   (set H or C)", RED, fs=8.5)
    arrow_h(axf, -L / 2, L / 2, -110, f"L  LENGTH {L:.0f}", RED, bold=True)
    if detailed:
        axf.add_patch(Rectangle((130, 0), 600, 870, fill=False, ec=BLUE, ls="--", lw=1.4))
        axf.text(430, 885, "typical built-in dishwasher\n600 W x 820-870 H (check yours)\n-> needs H = 915 to 965",
                 color=BLUE, fontsize=8, ha="center", va="bottom")
        axf.text(-430, under + 30, "rails 780 / 1640: calculated", color=GREY, fontsize=7.5, ha="center")
        axf.text(L / 2 + 35, 300, f"front legs\n= H - 180\n({args.leg * 1000:.0f} now)", color=GREY, fontsize=7.5)
        axf.text(-L / 2 + 120, 900, f"back uprights = H + B - 215 ({args.back_leg * 1000:.0f} now)", color=GREY, fontsize=7.5)
    axf.set_xlim(-L / 2 - 580, L / 2 + 350)
    axf.set_ylim(-200, zmax + 230)
    axf.set_title("FRONT VIEW", loc="left", fontsize=11)

    # ---- side view ----
    draw_view(axs, parts, 1, 2, 0, False, args.back_plate_radius * 1000)
    axs.plot([-D / 2 - 80, D / 2 + 80], [0, 0], color="k", lw=1)
    arrow_h(axs, -D / 2, D / 2, -110, f"D  DEPTH {D:.0f}", RED, bold=True)
    arrow_h(axs, D / 2 - T, D / 2, 450, f"T  table-top depth {T:.0f}", RED, fs=8.5, dz=14)
    if detailed:
        axs.text(-60, 1080, "brace joints follow H\n(340/380 above the rail)", color=GREY, fontsize=7.5)
        axs.text(0, 250, "lower rails stay at\n40-120 above leg bottom", color=GREY, fontsize=7.5, ha="center")
    axs.set_xlim(-D / 2 - 120, D / 2 + 140)
    axs.set_ylim(-200, zmax + 230)
    axs.set_title("SIDE VIEW", loc="left", fontsize=11)

    # ---- settings list ----
    txt = fig.add_axes([0.74, 0.06, 0.25, 0.86])
    txt.axis("off")
    lines = [
        ("WHAT YOU CAN CONFIGURE (mm)", RED, 12, "bold"),
        (f"H  table top height       {H:5.0f}", RED, 10, "bold"),
        ("   floor -> top surface", RED, 8.5, "normal"),
        (f"C  free height under top  {C:5.0f}", RED, 10, "bold"),
        ("   floor -> underside of the frame;", RED, 8.5, "normal"),
        ("   give H or C, not both", RED, 8.5, "normal"),
        (f"L  length                 {L:5.0f}", RED, 10, "bold"),
        (f"D  depth                  {D:5.0f}", RED, 10, "bold"),
        (f"B  back wall above top    {BW:5.0f}", RED, 10, "normal"),
        (f"T  table-top depth        {T:5.0f}", RED, 10, "normal"),
        ("   (top starts at the front edge)", RED, 8.5, "normal"),
        ("R  robot position on the top", RED, 10, "normal"),
        ("   x, y, angle from the table centre", RED, 8.5, "normal"),
        ("", "k", 6, "normal"),
        ("ON / OFF", RED, 11, "bold"),
        ("back wall, diagonal braces,", RED, 9, "normal"),
        ("middle legs, lower rails", RED, 9, "normal"),
    ]
    if detailed:
        lines += [
            ("", "k", 6, "normal"),
            ("CALCULATED (never typed in)", GREY, 11, "bold"),
            ("front legs     = H - 85 foot - 80 rail - 15 top", GREY, 8.5, "normal"),
            ("back uprights  = H + B - 215", GREY, 8.5, "normal"),
            ("rail lengths   1640, 780, 880, 800", GREY, 8.5, "normal"),
            ("planks, brace + notch positions", GREY, 8.5, "normal"),
            ("", "k", 6, "normal"),
            ("FIXED PARTS (measured, rarely change)", "#333", 11, "bold"),
            ("profile 80x80, feet 85, planks 120 x 15,", "#333", 8.5, "normal"),
            ("slots 10 / 30, back sheet 10 R20, brace 40x40,", "#333", 8.5, "normal"),
            ("brackets 80, adapter plate Ø350", "#333", 8.5, "normal"),
            ("", "k", 6, "normal"),
            ("CHECKS the script prints", BLUE, 11, "bold"),
            ("free height/width per bay vs. an appliance", BLUE, 8.5, "normal"),
            ("(e.g. dishwasher 600 x 870 x 570), overlaps,", BLUE, 8.5, "normal"),
            ("cut list of profile lengths to order", BLUE, 8.5, "normal"),
        ]
    y = 0.97
    for s, c, fs, w in lines:
        txt.text(0, y, s, color=c, fontsize=fs, fontweight=w, family="monospace" if fs < 11 else None, va="top")
        y -= 0.036 if fs >= 9.5 else 0.03
    title = ("Aluprofil table - proposed settings (mm). Red = you set it, grey = calculated from it."
             if detailed else "Aluprofil table - what you can configure (mm, current values)")
    fig.suptitle(title, x=0.02, ha="left", fontsize=13)
    base = os.path.join(DRAWINGS, ("config_proposal" if detailed else f"alu_table{type_tag(args)}_settings") + args.suffix)
    for ext in ("pdf", "png"):
        fig.savefig(f"{base}.{ext}", dpi=120 if ext == "png" else None)
    plt.close(fig)
    print(f"[INFO] Wrote {base}.pdf and .png")


if __name__ == "__main__":
    main()
