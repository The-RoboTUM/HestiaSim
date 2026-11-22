# HestiaSimulation (RoboTUM), author: Jeffrey

"""Generate a table built from aluminium T-slot profiles as a mesh file (meters, Z-up).

The profile cross-section (outer square, T-slots on a 40 mm grid, bores) is extruded with trimesh,
cut to length, and assembled from the floor up into:

    levelling feet -> 4 legs (2 short front legs, 2 tall back uprights) -> top rails resting on
    the legs -> slotted table-top planks (+ crossbar and vertical sheet on the back uprights,
    optional lower rails between the legs)

Defaults = the real lab table as measured on 2026-11-22: 1800 x 960 frame of 80x80 profiles,
85 mm feet, 3 x 500 mm front legs (corners + middle), 3 x 1320 mm back uprights (corners + middle), top 1800 x 840 of 7 x 120 mm planks
(15 mm thick, 10 mm slots every 30 mm, M6 T-nuts) flush with the front -> top at 680 mm,
back sheet 1800 x 420 x 10 (R20) sticking out 50 mm above the crossbar -> highest point 1535 mm.

.. code-block:: bash

    uv run --with mapbox-earcut python ~/aluprofil_table/make_table.py
    uv run --with mapbox-earcut python ~/aluprofil_table/make_table.py --lower-rail-z 0  # no lower rails
"""

import argparse
import os

import numpy as np
import trimesh
from paths import MESHES
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union


def profile_section(
    size: float = 0.080,
    slot: float = 0.0082,
    lip: float = 0.0045,
    chamber: float = 0.018,
    chamber_depth: float = 0.0055,
    bore: float = 0.0085,
) -> Polygon:
    """Cross-section of a square T-slot profile centered at the origin (meters).

    Slots sit on a 40 mm grid, so a 40x40 profile has one slot per side and an 80x80 has two.
    """
    h = size / 2.0
    n = max(1, round(size / 0.04))
    offsets = [(-h + 0.02 + 0.04 * k) for k in range(n)]  # slot centers along a side
    cutouts = []
    for angle in (0.0, 90.0, 180.0, 270.0):
        c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        for o in offsets:
            # slot on the +y side, then rotated: opening through the lip, T-chamber behind it
            opening = box(o - slot / 2, h - lip, o + slot / 2, h + 1e-4)
            tee = box(o - chamber / 2, h - lip - chamber_depth, o + chamber / 2, h - lip)
            shape = unary_union([opening, tee])
            cutouts.append(Polygon([(c * x - s * y, s * x + c * y) for x, y in shape.exterior.coords]))
    # one bore per 40 mm cell
    for ox in offsets:
        for oy in offsets:
            cutouts.append(Point(ox, oy).buffer(bore / 2, resolution=12))
    section = box(-h, -h, h, h).difference(unary_union(cutouts))
    assert section.is_valid and section.geom_type == "Polygon", "invalid profile section"
    return section


def profile_bar(section: Polygon, length: float, axis: str, center: tuple[float, float, float]) -> trimesh.Trimesh:
    """Extrude the section to ``length`` along ``axis`` ('x', 'y' or 'z') and place its center."""
    bar = trimesh.creation.extrude_polygon(section, length)  # along +z, from 0..length
    bar.apply_translation((0, 0, -length / 2))
    if axis == "x":
        bar.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0)))
    elif axis == "y":
        bar.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0)))
    bar.apply_translation(center)
    return bar


def profile_between(section: Polygon, p0, p1) -> trimesh.Trimesh:
    """Extrude ``section`` along the straight line from point ``p0`` to ``p1``.

    The section's x axis is kept as close to world x as possible, so a bar in the yz plane has its
    flat faces facing +-x (not rotated 45 deg about its own axis).
    """
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    v = p1 - p0
    length = float(np.linalg.norm(v))
    ez = v / length
    ref = np.array([1.0, 0, 0]) if abs(ez[0]) < 0.9 else np.array([0, 1.0, 0])
    ex = ref - ez * ez.dot(ref)
    ex /= np.linalg.norm(ex)
    frame = np.eye(4)
    frame[:3, :3] = np.column_stack((ex, np.cross(ez, ex), ez))
    bar = trimesh.creation.extrude_polygon(section, length)
    bar.apply_translation((0, 0, -length / 2))
    bar.apply_transform(frame)
    bar.apply_translation((p0 + p1) / 2)
    return bar


def corner_wedge(corner, along, down: float, out: float, thick: float) -> trimesh.Trimesh:
    """Solid right-angle wedge for an angle bracket (Winkel) in a leg/rail corner.

    ``corner`` = point where the leg face meets the rail underside (centre of the bracket width),
    ``along`` = horizontal unit vector from the leg along the rail. Arms: ``down`` along the leg,
    ``out`` along the rail; ``thick`` = bracket width across.
    """
    c, a, z = np.asarray(corner, float), np.asarray(along, float), np.array([0, 0, 1.0])
    w = np.cross(a, z) * thick / 2
    tri = [c, c + a * out, c - z * down]
    verts = [p + w for p in tri] + [p - w for p in tri]
    faces = [[0, 1, 2], [3, 5, 4], [0, 3, 4], [0, 4, 1], [1, 4, 5], [1, 5, 2], [2, 5, 3], [2, 3, 0]]
    mesh = trimesh.Trimesh(verts, faces, process=False)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def plate_section(
    width: float,
    thickness: float,
    slot: float = 0.0082,
    lip: float = 0.0045,
    chamber: float = 0.018,
    chamber_depth: float = 0.0055,
    wall: float = 0.003,
    hollow: bool = False,
    pitch: float = 0.030,
) -> Polygon:
    """Cross-section of one table-top plank (x = across the plank, y = up), centered at the origin.

    Solid plank with T-slots on the top face every ``pitch``. The real table (measured 2026-11-22):
    120 mm planks, 10 mm slot opening, slots every 30 mm (4 per plank), M6 T-nuts (4.7 mm measured
    bore). Lip, T-chamber width and depth are NOT measured: they are generic placeholder values,
    only visual (MuJoCo collides the hull; the slots don't change where things rest on the top).
    Correction (2026-11-22): earlier notes said these planks were slot 8 on a 40 mm grid taking
    M8 nuts, like item's "Profile 8 160x16". That was wrong: the measured slots are 10 mm / 30 mm
    and the nuts are M6; the product is not identified.
    Use pitch=0.045 for Vention table-top extrusions. ``hollow=True`` adds inner chambers.
    """
    hw, ht = width / 2.0, thickness / 2.0
    n = max(1, round(width / pitch))
    offsets = [(k - (n - 1) / 2) * pitch for k in range(n)]  # slots centered on the plank
    cutouts = []
    slot_bottom = ht - lip - chamber_depth
    for o in offsets:
        if slot_bottom > -ht + wall:
            cutouts.append(box(o - slot / 2, ht - lip, o + slot / 2, ht + 1e-4))
            cutouts.append(box(o - chamber / 2, slot_bottom, o + chamber / 2, ht - lip))
    section = box(-hw, -ht, hw, ht).difference(unary_union(cutouts))
    # hollow chambers below the slots, one per 40 mm cell, separated by webs
    holes = []
    y0, y1 = -ht + wall, slot_bottom - wall
    if hollow and y1 - y0 > 0.006:
        for o in offsets:
            holes.append(box(o - 0.015, y0, o + 0.015, y1))
        section = section.difference(unary_union(holes))
    assert section.is_valid and section.geom_type == "Polygon", "invalid plate section"
    return section


def profile_plate(
    length: float,
    width: float,
    thickness: float,
    plank: float,
    along: str,
    z_bottom: float,
    pitch: float = 0.030,
    slot: float = 0.010,
    notches: list[tuple[float, float, float, float]] = (),
) -> dict[str, trimesh.Trimesh]:
    """Table top of ``length`` (x) x ``width`` (y) made of planks ``plank`` wide running along ``along``.

    ``notches`` = rectangles (x0, x1, y0, y1) in plate coordinates (centered on the plate) cut out
    of the planks, e.g. to let a brace reach the frame rail below. Only for planks along x.

    As many full-width planks as fit; if the span is not a multiple of ``plank``, one narrower plank
    fills the rest at the low-coordinate end (-y for planks along x = the back when the top is
    flush with +y). E.g. 880 mm with 160 mm planks -> 80 + 5 x 160.
    """
    run, span = (length, width) if along == "x" else (width, length)
    n_full = max(1, int(span / plank + 1e-6))
    rest = span - n_full * plank
    widths = ([rest] if rest > 1e-3 else []) + [min(plank, span)] * n_full
    if along == "x":  # section x -> world y, section y -> world z, extrusion -> world x
        rot = [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    else:  # section x -> world x, section y -> world z, extrusion -> world -y
        rot = [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    parts = {}
    start = -span / 2
    for k, w in enumerate(widths):
        c = start + w / 2
        start += w
        full = plate_section(w, thickness, slot=slot, pitch=pitch)  # section in xy, extruded along +z
        cuts = [n for n in notches if along == "x" and n[2] < c + w / 2 - 1e-6 and n[3] > c - w / 2 + 1e-6]
        xs = sorted({-run / 2, run / 2, *(min(max(v, -run / 2), run / 2) for n in cuts for v in n[:2])})
        pieces = []
        for a, b in zip(xs[:-1], xs[1:]):  # extrude segment by segment, notched section where cut
            if b - a < 1e-6:
                continue
            sec = full
            for n in cuts:
                if n[0] < (a + b) / 2 < n[1]:
                    sec = sec.difference(box(n[2] - c, -thickness, n[3] - c, thickness))
            for poly in [sec] if sec.geom_type == "Polygon" else list(getattr(sec, "geoms", [])):
                if poly.area > 1e-8:
                    piece = trimesh.creation.extrude_polygon(poly, b - a)
                    piece.apply_translation((0, 0, a))
                    pieces.append(piece)
        bar = trimesh.util.concatenate(pieces)
        bar.apply_transform(np.array(rot, dtype=float))
        offset = (0, c, 0) if along == "x" else (c, 0, 0)
        bar.apply_translation((offset[0], offset[1], z_bottom + thickness / 2))
        parts[f"plate_plank_{k}"] = bar
    return parts


def levelling_foot(height: float, size: float, x: float, y: float) -> list[trimesh.Trimesh]:
    """Round pad on the floor + threaded stem + hex nut up to the leg bottom."""
    pad_h = min(0.02, height * 0.3)
    pad = trimesh.creation.cylinder(radius=min(0.04, size * 0.6), height=pad_h, sections=32)
    pad.apply_translation((x, y, pad_h / 2))
    nut_h = min(0.01, height * 0.15)
    stem = trimesh.creation.cylinder(radius=0.008, height=height - pad_h - nut_h, sections=16)
    stem.apply_translation((x, y, pad_h + (height - pad_h - nut_h) / 2))
    nut = trimesh.creation.cylinder(radius=0.015, height=nut_h, sections=6)
    nut.apply_translation((x, y, height - nut_h / 2))
    return [pad, stem, nut]


def build_table(
    length: float,
    width: float,
    size: float,
    foot: float,
    leg: float,
    plate: float,
    lower_rail_z: float,
    plate_width: float,
    plank: float,
    plank_along: str = "y",
    plate_side: str = "center",
    back_leg: float = 0.0,
    back_plate_h: float = 0.0,
    back_plate_t: float = 0.0,
    back_plate_over: float = 0.0,
    back_plate_r: float = 0.0,
    plank_pitch: float = 0.030,
    plank_slot: float = 0.010,
    back_posts: int = 2,
    back_plate_rails: bool = False,
    lower_rail_sides: str = "all",
    mid_rail: bool = False,
    brace_size: float = 0.0,
    brace_h: float = 0.34,
    brace_v: float = 0.34,
    brace_clearance: float = 0.02,
    front_posts: int = 2,
    lower_rail_mid: bool = False,
    bracket_leg: float = 0.0,
    bracket_rail: float = 0.0,
) -> dict[str, trimesh.Trimesh]:
    """Build the table parts. Origin is on the floor, at the table center.

    ``back_posts`` = number of tall back uprights (2 = corners only; 3 = one more in the middle,
    as on the real table). Extra uprights are evenly spaced and the back rail at table height is
    split between them, e.g. 1800 mm with 3 posts -> 80 + 780 + 80 + 780 + 80.
    """
    section = profile_section(size)
    h = size / 2
    leg_top = foot + leg
    rail_z = leg_top + h  # top rails rest on the legs
    frame_top = leg_top + size
    parts: dict[str, trimesh.Trimesh] = {}
    corners = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    # back side = the long side the top is NOT flush with; its legs can be taller uprights
    back = {"+y": -1, "-y": 1}.get(plate_side, 0)
    tall_back = back != 0 and back_leg > leg
    for i, (sx, sy) in enumerate(corners):
        x, y = sx * (length / 2 - h), sy * (width / 2 - h)
        l = back_leg if (tall_back and sy == back) else leg
        parts[f"leg_{i}"] = profile_bar(section, l, "z", (x, y, foot + l / 2))
        if foot > 0:
            for j, m in enumerate(levelling_foot(foot, size, x, y)):
                parts[f"foot_{i}_{j}"] = m
    n_posts = max(2, back_posts) if tall_back else 2
    post_x = [-length / 2 + h + k * (length - size) / (n_posts - 1) for k in range(n_posts)]
    for k, x in enumerate(post_x[1:-1]):  # middle back uprights, same build as the corner ones
        y = back * (width / 2 - h)
        parts[f"leg_mid_{k}"] = profile_bar(section, back_leg, "z", (x, y, foot + back_leg / 2))
        if foot > 0:
            for j, m in enumerate(levelling_foot(foot, size, x, y)):
                parts[f"foot_mid_{k}_{j}"] = m
    # middle front legs (same 500 mm leg + foot as the corners), under the front rail, which runs
    # over them unbroken
    front_mid_x = []
    if back != 0 and front_posts > 2:
        y = -back * (width / 2 - h)
        for k in range(1, front_posts - 1):
            x = -length / 2 + h + k * (length - size) / (front_posts - 1)
            front_mid_x.append(x)
            parts[f"leg_front_mid_{k - 1}"] = profile_bar(section, leg, "z", (x, y, foot + leg / 2))
            if foot > 0:
                for j, m in enumerate(levelling_foot(foot, size, x, y)):
                    parts[f"foot_front_mid_{k - 1}_{j}"] = m
    # top rails on the legs. Without tall back uprights: long rails full length, short rails
    # between them. With tall back uprights (as measured on the real table, 2026-11-22): the side
    # rails run from the uprights' inner face to the front edge (width - size), and the front and
    # back rails both fit between them / between the uprights (length - 2 * size); with middle
    # back uprights the back rail is split into one piece per gap.
    for sy in (1, -1):
        if tall_back and sy == back and n_posts > 2:
            seg = (length - n_posts * size) / (n_posts - 1)
            for k in range(n_posts - 1):
                parts[f"top_rail_x_{'p' if sy > 0 else 'n'}_{k}"] = profile_bar(
                    section, seg, "x", ((post_x[k] + post_x[k + 1]) / 2, sy * (width / 2 - h), rail_z)
                )
            continue
        rail_len = length - 2 * size if tall_back else length
        parts[f"top_rail_x_{'p' if sy > 0 else 'n'}"] = profile_bar(
            section, rail_len, "x", (0, sy * (width / 2 - h), rail_z)
        )
    if tall_back:
        # horizontal crossbar resting on top of the two back uprights, full length
        parts["back_crossbar"] = profile_bar(
            section, length, "x", (0, back * (width / 2 - h), foot + back_leg + h)
        )
    if tall_back and back_plate_h > 0 and back_plate_t > 0:
        # flat vertical aluminium sheet on the inner face of the back uprights;
        # top edge sticks out back_plate_over above the crossbar
        top_z = foot + back_leg + size + back_plate_over
        r = min(back_plate_r, back_plate_h / 2 - 1e-4, length / 2 - 1e-4)
        outline = box(-length / 2, -back_plate_h / 2, length / 2, back_plate_h / 2)
        if r > 0:  # rounded corners
            outline = outline.buffer(-r).buffer(r, resolution=16)
        sheet = trimesh.creation.extrude_polygon(outline, back_plate_t)
        sheet.apply_translation((0, 0, -back_plate_t / 2))
        # outline x -> world x, outline y -> world z, thickness -> world y
        sheet.apply_transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float))
        sheet.apply_translation((0, back * (width / 2 - size - back_plate_t / 2), top_z - back_plate_h / 2))
        parts["back_plate"] = sheet
        if back_plate_rails:
            # 80x80 rails between the back uprights carrying the sheet, hidden behind it (in the
            # uprights' plane); their bottom face is level with the sheet's bottom edge, no offset
            seg = (length - n_posts * size) / (n_posts - 1)
            z = top_z - back_plate_h + h
            for k in range(n_posts - 1):
                parts[f"back_plate_rail_{k}"] = profile_bar(
                    section, seg, "x", ((post_x[k] + post_x[k + 1]) / 2, back * (width / 2 - h), z)
                )
    section = profile_section(size)
    side_len, side_y = (width - size, -back * h) if tall_back else (width - 2 * size, 0.0)
    for sx in (1, -1):
        parts[f"top_rail_y_{'p' if sx > 0 else 'n'}"] = profile_bar(
            section, side_len, "y", (sx * (length / 2 - h), side_y, rail_z)
        )
    if mid_rail:
        # cross rail under the middle of the top (x = 0), between the front and back rails
        parts["top_rail_y_mid"] = profile_bar(section, width - 2 * size, "y", (0, 0, rail_z))
    if bracket_rail > 0 and bracket_leg > 0:
        # angle brackets (Winkel) in the inner corner between each corner leg/upright and the long
        # rail resting on / between them (front and back). Modelled as a solid wedge: the L plus its
        # triangular side walls, no holes or screws. Width = profile face.
        rail_bottom = leg_top
        for sy in (1, -1):
            for sx in (1, -1):  # corner legs -> long front / back rail, towards the table centre
                corner = (sx * (length / 2 - size), sy * (width / 2 - h), rail_bottom)
                parts[f"bracket_{'front' if sy == -back else 'back'}_{'p' if sx > 0 else 'n'}"] = corner_wedge(
                    corner, (-sx, 0, 0), bracket_leg, bracket_rail, size
                )
        if mid_rail and back != 0:
            # middle front leg -> middle cross rail under the top (towards the back)
            for k, x in enumerate(front_mid_x):
                if abs(x) < 1e-3:
                    corner = (x, -back * (width / 2 - size), rail_bottom)
                    parts[f"bracket_front_mid_{k}"] = corner_wedge(corner, (0, back, 0), bracket_leg, bracket_rail, size)
        if tall_back:
            # middle back uprights -> back rail pieces on both sides (not to the middle cross rail)
            for k, px in enumerate(post_x[1:-1]):
                for sx in (1, -1):
                    corner = (px + sx * h, back * (width / 2 - h), rail_bottom)
                    parts[f"bracket_back_mid_{k}_{'p' if sx > 0 else 'n'}"] = corner_wedge(
                        corner, (sx, 0, 0), bracket_leg, bracket_rail, size
                    )
    # optional lower rails butt between the legs: "y" = left/right sides only (real table),
    # "x" = front/back only, "all" = all four. A back x rail is split at middle back uprights.
    if lower_rail_z > 0 and lower_rail_sides in ("x", "all"):
        for sy in (1, -1):
            tag = f"lower_rail_x_{'p' if sy > 0 else 'n'}"
            if tall_back and sy == back and n_posts > 2:
                seg = (length - n_posts * size) / (n_posts - 1)
                for k in range(n_posts - 1):
                    parts[f"{tag}_{k}"] = profile_bar(
                        section, seg, "x", ((post_x[k] + post_x[k + 1]) / 2, sy * (width / 2 - h), lower_rail_z)
                    )
                continue
            parts[tag] = profile_bar(section, length - 2 * size, "x", (0, sy * (width / 2 - h), lower_rail_z))
    if lower_rail_z > 0 and lower_rail_mid:
        # lower rail across the width at each middle front leg that has a back upright opposite it
        # (real table: x = 0), same height as the side lower rails
        for k, x in enumerate(front_mid_x):
            if tall_back and any(abs(x - px) < 1e-3 for px in post_x):
                parts[f"lower_rail_y_mid_{k}"] = profile_bar(section, width - 2 * size, "y", (x, 0, lower_rail_z))
    if lower_rail_z > 0 and lower_rail_sides in ("y", "all"):
        for sx in (1, -1):
            parts[f"lower_rail_y_{'p' if sx > 0 else 'n'}"] = profile_bar(
                section, width - 2 * size, "y", (sx * (length / 2 - h), 0, lower_rail_z)
            )
    # flush with one long edge of the frame (+y / -y) or centered
    shift = {"+y": 1, "-y": -1, "center": 0}[plate_side] * (width - plate_width) / 2
    notches = []
    if tall_back and brace_size > 0:
        # diagonal brace per side: back corner upright -> side rail at table height, held by a swivel
        # joint (Gelenkverbinder, modelled as a brace_size cube) at each end. Measured to the joints:
        # lower joint 340..380 mm in front of the upright's front face, on the side rail's top face;
        # upper joint 340..380 mm above the side rail's top face, on the upright's front face.
        # Sideways the brace sits on the OUTER half of the 80x80 profiles (flush with the outer
        # face), not on their centre line.
        s2, d = brace_size / 2, -back  # d = direction from back to front
        y_face = back * (width / 2 - size)
        brace_section = profile_section(brace_size)
        for sx in (1, -1):
            tag = "p" if sx > 0 else "n"
            x = sx * (length / 2 - s2)
            lo = (x, y_face + d * (brace_h + s2), frame_top + s2)
            hi = (x, y_face + d * s2, frame_top + brace_v + s2)
            for name, p in (("lo", lo), ("hi", hi)):
                joint = trimesh.creation.box(extents=(brace_size, brace_size, brace_size))
                joint.apply_translation(p)
                parts[f"brace_joint_{name}_{tag}"] = joint
            parts[f"brace_{tag}"] = profile_between(brace_section, lo, hi)
            # plank cut-out over the rail so the lower joint sits on the frame, not on the top
            # (joint + clearance; clipped at the plank ends)
            y0, y1 = sorted((lo[1] - s2 - brace_clearance, lo[1] + s2 + brace_clearance))
            x0, x1 = x - s2 - brace_clearance, x + s2 + brace_clearance
            notches.append((x0, x1, y0 - shift, y1 - shift))
    if plate > 0:
        plate_parts = profile_plate(
            length, plate_width, plate, plank, plank_along, frame_top, plank_pitch, plank_slot, notches
        )
        for m in plate_parts.values():
            m.apply_translation((0, shift, 0))
        parts.update(plate_parts)
    return parts


# Table types: named variants of the same table. A type sets some options and wins over the flags.
TYPES = {
    "standard": {},  # the measured lab table, with the two diagonal braces
    "no_brace": {"brace_size": 0.0},  # same table without the diagonal braces (so also without their plank notches)
}


def type_tag(args: argparse.Namespace) -> str:
    """'' for the standard table, '_<type>' otherwise; goes into every output file name."""
    return "" if args.type == "standard" else f"_{args.type}"


def build_from_args(args: argparse.Namespace) -> dict[str, trimesh.Trimesh]:
    """Build the table parts from parsed command-line arguments (after applying the table type)."""
    for key, value in TYPES[args.type].items():
        setattr(args, key, value)
    return build_table(
        args.length,
        args.width,
        args.profile,
        args.foot,
        args.leg,
        args.plate,
        args.lower_rail_z,
        args.plate_width,
        args.plank,
        args.plank_along,
        args.plate_side,
        args.back_leg,
        args.back_plate_height,
        args.back_plate_thickness,
        args.back_plate_over,
        args.back_plate_radius,
        args.plank_pitch,
        args.plank_slot,
        args.back_posts,
        not args.no_back_plate_rails,
        args.lower_rail_sides,
        not args.no_mid_rail,
        args.brace_size,
        args.brace_h,
        args.brace_v,
        front_posts=args.front_posts,
        lower_rail_mid=not args.no_lower_rail_mid,
        bracket_leg=args.bracket_leg,
        bracket_rail=args.bracket_rail,
    )


def build_parser() -> argparse.ArgumentParser:
    """Command-line options shared by make_table.py and make_drawing.py."""
    parser = argparse.ArgumentParser(description="Generate an aluminium-profile table mesh.")
    parser.add_argument("--type", choices=sorted(TYPES), default="standard",
                        help="Table type: standard (with diagonal braces) or no_brace.")
    parser.add_argument("--length", type=float, default=1.80, help="Table length along x [m].")
    parser.add_argument("--width", type=float, default=0.96, help="Table width along y [m].")
    parser.add_argument("--profile", type=float, default=0.08, help="Profile size [m] (0.04, 0.08, ...).")
    parser.add_argument("--foot", type=float, default=0.085, help="Levelling foot height [m], 0 to disable.")
    parser.add_argument("--leg", type=float, default=0.50, help="Leg profile length [m].")
    parser.add_argument(
        "--back-leg",
        type=float,
        default=1.32,
        help="Length of the two back uprights [m] (back = side opposite --plate-side); <= --leg disables.",
    )
    parser.add_argument(
        "--back-posts", type=int, default=3, help="Number of tall back uprights (2 = corners, 3 = + one in the middle)."
    )
    parser.add_argument(
        "--front-posts", type=int, default=3, help="Number of front legs (2 = corners, 3 = + one in the middle)."
    )
    parser.add_argument(
        "--no-back-plate-rails",
        action="store_true",
        help="Leave out the rails between the back uprights that carry the sheet (bottom-flush with it).",
    )
    parser.add_argument("--back-plate-height", type=float, default=0.42, help="Back vertical plate height [m], 0 = none.")
    parser.add_argument("--back-plate-thickness", type=float, default=0.01, help="Back vertical sheet thickness [m].")
    parser.add_argument(
        "--back-plate-over", type=float, default=0.05, help="How far the back plate sticks out above the crossbar [m]."
    )
    parser.add_argument("--back-plate-radius", type=float, default=0.02, help="Back sheet corner radius [m].")
    parser.add_argument("--plate", type=float, default=0.015, help="Table-top plank thickness [m], 0 to disable.")
    parser.add_argument("--plate-width", type=float, default=0.84, help="Table-top total width (y) [m].")
    parser.add_argument(
        "--plate-side", choices=("+y", "-y", "center"), default="+y", help="Long edge the top is flush with."
    )
    parser.add_argument("--plank", type=float, default=0.12, help="Width of one full table-top plank [m]; a narrower one fills the rest.")
    parser.add_argument(
        "--plank-pitch", type=float, default=0.03, help="T-slot spacing on the planks, centre to centre [m]."
    )
    parser.add_argument(
        "--plank-slot", type=float, default=0.010, help="T-slot opening width on the planks [m]."
    )
    parser.add_argument(
        "--plank-along", choices=("x", "y"), default="x", help="Direction the planks run (x = along the length)."
    )
    parser.add_argument(
        "--lower-rail-z",
        type=float,
        default=0.165,
        help="Center height of lower rails above the floor [m], 0 = none. Real table: bottom 40 mm above"
        " the leg profiles' bottom (85 mm feet) -> 0.085 + 0.04 + 0.04 = 0.165.",
    )
    parser.add_argument(
        "--no-mid-rail", action="store_true", help="Leave out the cross rail under the middle of the table top."
    )
    parser.add_argument(
        "--brace-size", type=float, default=0.04, help="Diagonal brace profile size [m] (40x40), 0 = no braces."
    )
    parser.add_argument(
        "--brace-h", type=float, default=0.34,
        help="Lower brace joint: distance from the back upright's front face to the joint [m] (joint = 340..380 mm).",
    )
    parser.add_argument(
        "--brace-v", type=float, default=0.34,
        help="Upper brace joint: height above the side rail's top face to the joint [m] (joint = 340..380 mm).",
    )
    parser.add_argument(
        "--no-lower-rail-mid",
        action="store_true",
        help="Leave out the lower rail between the middle front leg and the middle back upright.",
    )
    parser.add_argument(
        "--bracket-leg", type=float, default=0.08,
        help="Corner angle brackets: arm length down the leg [m] (PLACEHOLDER, not measured), 0 = none.",
    )
    parser.add_argument(
        "--bracket-rail", type=float, default=0.08,
        help="Corner angle brackets: arm length along the long rail [m] (PLACEHOLDER, not measured), 0 = none.",
    )
    parser.add_argument(
        "--lower-rail-sides", choices=("y", "x", "all"), default="y", help="Which sides get lower rails (y = left/right)."
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output mesh (default: meshes/alu_table[_<type>].obj)."
    )
    return parser


def main():
    args = build_parser().parse_args()
    parts = build_from_args(args)
    if args.output is None:
        args.output = os.path.join(MESHES, f"alu_table{type_tag(args)}.obj")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    table = trimesh.util.concatenate(list(parts.values()))
    table.export(args.output)
    print(f"[INFO] {len(parts)} parts, {len(table.faces)} faces, bounds [m]:\n{table.bounds}")
    if "plate_plank_0" in parts:
        print(f"[INFO] table top at z = {parts['plate_plank_0'].bounds[1][2]:.4f} m")
    print(f"[INFO] highest point z = {table.bounds[1][2]:.4f} m")
    print(f"[INFO] Wrote {args.output}")


if __name__ == "__main__":
    main()
