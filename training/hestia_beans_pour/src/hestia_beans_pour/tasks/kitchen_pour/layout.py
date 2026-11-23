# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hestia kitchen geometry for the kitchen beans-pour task (pure Python, no Isaac imports).

Every table, robot and mixer position comes from ``scenes/u_table_layout/u_layout.py`` (the single source of truth of
the U table layout), so the training always matches the scene. This module only adds what the task needs on top:

* the bean container (``components/container``, container01 v2) as box colliders, and where it stands on the middle
  (bottom) table,
* simplified collision boxes of the mixer (the mixer USD's convex-decomposition colliders would close the chute hole),
* the regions that count beans as delivered into the mixer or spilled,
* generated USD files: box colliders for the three tables and the Panda adapter plate, and the container.

Frames
------
*Env frame* = the U-layout world frame: origin on the floor at the centre of the bottom table, metres, Z-up.
*Container frame* = the frame of ``components/container/container01_v2.stl`` (and of the current v3, same frame; converted from mm to m): origin
at the centre of the bottom face of its bounding box (handle included), ``+x`` = the side with the handle.
*Mixer frame* = the root of the mixer USD (prim ``mum8``): 0.233 m above the mixer's bottom, so ``z = 0`` is the top
of the lid plate; ``+x`` = the mixer's front (bowl / chute side). All mixer numbers below were measured from
``components/mixer/mum8_chute_v2.usda`` (mixer v2, Mana) and ``mum8_v10_vs_original.png``.
Quaternions handed to Isaac Lab are ``(x, y, z, w)`` (Isaac Lab 3.x convention).
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

##
# Repository access
##


def hestia_sim_root() -> Path:
    """Root of the HestiaSim repository (override with the ``HESTIA_SIM_ROOT`` environment variable)."""
    configured = os.environ.get("HESTIA_SIM_ROOT")
    # .../HestiaSim/training/hestia_beans_pour/src/hestia_beans_pour/tasks/kitchen_pour/layout.py
    root = Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[6]
    if not (root / "scenes" / "u_table_layout" / "u_layout.py").is_file():
        raise FileNotFoundError(
            f"HestiaSim repository not found at {root} (expected scenes/u_table_layout/u_layout.py). The kitchen task "
            "reads the U table layout from the repository; install the project editable (uv sync does) or set "
            "HESTIA_SIM_ROOT=<path to HestiaSim>."
        )
    return root


@lru_cache(maxsize=1)
def u_layout():
    """The scene's ``u_layout`` module, loaded from its file (no ``sys.path`` changes)."""
    path = hestia_sim_root() / "scenes" / "u_table_layout" / "u_layout.py"
    spec = importlib.util.spec_from_file_location("hestia_u_layout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quat_xyzw_from_yaw(yaw_deg: float) -> tuple[float, float, float, float]:
    """Rotation about +z as an Isaac Lab ``(x, y, z, w)`` quaternion."""
    half = math.radians(yaw_deg) / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


##
# Bean container (components/container, measured on container01_v2.stl), in the container frame [m]
##
# Measured from the STL (sections through the mesh): an open rectangular box, 160 x 120 mm outside, 69.95 mm high,
# 5 mm walls and a 5 mm floor (corners rounded with ~4 mm, ignored here), plus a 20 mm wide hook-shaped handle on
# the +x side: a top bar flush with the rim (x 65..95, z 55..69.95) and a leg hanging down at its outer end
# (x 80..95, down to z ~10, rounded at the bottom). Between the leg and the box wall is a 15 mm slot, open at the
# bottom. The gripper (max. 80 mm) cannot span the box, so the container is grasped at the handle.

CONTAINER_STL = "components/container/container01_v3.stl"  # current version (simplified v2, same shape within 0.03 mm)
CONTAINER_BODY_X = (-0.095, 0.065)
CONTAINER_HALF_Y = 0.060
CONTAINER_HEIGHT = 0.06995
CONTAINER_WALL = 0.005
CONTAINER_FLOOR = 0.005
HANDLE_HALF_Y = 0.010
HANDLE_BAR_X = (0.065, 0.095)
HANDLE_BAR_Z = (0.055, CONTAINER_HEIGHT)
HANDLE_LEG_X = (0.080, 0.095)
HANDLE_LEG_Z = (0.012, 0.055)

CONTAINER_CAVITY_LO = (CONTAINER_BODY_X[0] + CONTAINER_WALL, -CONTAINER_HALF_Y + CONTAINER_WALL, CONTAINER_FLOOR)
CONTAINER_CAVITY_HI = (CONTAINER_BODY_X[1] - CONTAINER_WALL, CONTAINER_HALF_Y - CONTAINER_WALL, CONTAINER_HEIGHT)
"""Inner cavity (x -0.090..0.060, y +-0.055, z 0.005..0.06995): 150 x 110 x 64.95 mm."""
CONTAINER_CAVITY_CENTRE_X = 0.5 * (CONTAINER_CAVITY_LO[0] + CONTAINER_CAVITY_HI[0])
"""The cavity is 15 mm off the frame origin (towards -x), because the handle sticks out on +x."""

HANDLE_GRASP_POINT = (0.0875, 0.0, 0.050)
"""Where the TCP holds the handle: middle of the leg, 20 mm below the rim (fingertips stay above the table)."""
HANDLE_GRASP_TCP_QUAT = (0.0, 1.0, 0.0, 0.0)
"""TCP orientation for the handle grasp, ``(x, y, z, w)`` in the container frame: 180 deg about y = gripper pointing
straight down, fingers closing along the container's y axis across the 20 mm handle. The grasp turned by 180 deg about
the TCP axis (fingers swapped) is equivalent."""
HANDLE_WIDTH = 2.0 * HANDLE_HALF_Y
"""Gripper width when holding the handle [m]."""


def container_rim_points() -> tuple[tuple[float, float, float], ...]:
    """Corners and edge midpoints of the cavity's rim, in the container frame (the possible pouring lips)."""
    (x0, y0, _), (x1, y1, z) = CONTAINER_CAVITY_LO, CONTAINER_CAVITY_HI
    xm = 0.5 * (x0 + x1)
    return ((x0, y0, z), (x0, y1, z), (x1, y0, z), (x1, y1, z), (x0, 0.0, z), (x1, 0.0, z), (xm, y0, z), (xm, y1, z))


def container_collider_boxes() -> tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...]:
    """``(name, centre, half extents)`` of the container's box colliders (floor, 4 walls, handle bar and leg)."""
    (bx0, bx1), hy, h = CONTAINER_BODY_X, CONTAINER_HALF_Y, CONTAINER_HEIGHT
    (cx0, cy0, _), (cx1, cy1, _) = CONTAINER_CAVITY_LO, CONTAINER_CAVITY_HI
    boxes = {
        "floor": _box((bx0, bx1), (-hy, hy), (0.0, CONTAINER_FLOOR)),
        "wall_px": _box((cx1, bx1), (-hy, hy), (CONTAINER_FLOOR, h)),
        "wall_nx": _box((bx0, cx0), (-hy, hy), (CONTAINER_FLOOR, h)),
        "wall_py": _box((cx0, cx1), (cy1, hy), (CONTAINER_FLOOR, h)),
        "wall_ny": _box((cx0, cx1), (-hy, cy0), (CONTAINER_FLOOR, h)),
        "handle_bar": _box(HANDLE_BAR_X, (-HANDLE_HALF_Y, HANDLE_HALF_Y), HANDLE_BAR_Z),
        "handle_leg": _box(HANDLE_LEG_X, (-HANDLE_HALF_Y, HANDLE_HALF_Y), HANDLE_LEG_Z),
    }
    return tuple((name, centre, half) for name, (centre, half, _) in boxes.items())


##
# Container placement on the middle table
##

SOURCE_TABLE = "TableBottom"
"""The middle table of the U (between the two side tables)."""
CUP_CENTRE_FROM_END = 0.15
"""Container frame origin (centre of its 190 x 120 mm footprint, handle included) -> the bottom table's +x end (the end
that touches the LEFT table, the robot / mixer side) [m]."""
CUP_CENTRE_FROM_FRONT = 0.20
"""Container frame origin -> the bottom table's front edge (the edge facing the inside of the U) [m]."""
CUP_YAW_DEG = 0.0
"""Container yaw relative to the bottom table: 0 = its sides parallel to the table edges, handle (+x) pointing to the
table's +x end, i.e. towards the left table and the robot."""

##
# Mixer geometry in the mixer frame (z = 0 at the top of the lid plate, +x = front)
##

MIXER_ROOT_LIFT = 0.233
"""Height of the mixer frame above the mixer's bottom [m] (``u_layout.MIXER_BOTTOM_OFFSET``)."""

# Chute (funnel on the lid): floor opening x 0.078..0.114, |y| <= ~0.040 at z = 0. Its back wall rises from
# (x 0.0777, z 0) to (0.060, 0.049), its front lip from (0.1147, 0) to (0.130, 0.0234); the funnel's top opening is
# the slanted plane z = 0.049 - 0.35 (x - 0.060) for x in 0.060..0.130, about 0.11 m wide.
CHUTE_HOLE_X = (0.078, 0.114)
CHUTE_HOLE_HALF_Y = 0.040
CHUTE_TOP_CENTRE = (0.095, 0.0, 0.037)
"""Centre of the funnel's (slanted) top opening [m]."""
POUR_TARGET_ABOVE_CHUTE = 0.045
"""The reward guides the container mouth to this height above :data:`CHUTE_TOP_CENTRE` [m]."""

# Collision boxes: (name, centre, half extents, rotation about the mixer's y axis [deg]).
# They replace the mixer USD's colliders (convex decomposition would close the chute hole and the bowl).


def _slanted_wall(p0, p1, thickness, half_y, into):
    """Box whose large face runs from p0 to p1 (x, z) and that extends ``thickness`` towards ``into`` (+1/-1 = the
    side of the +x / -x normal)."""
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dz)
    # normal of the face, oriented to the material side
    nx, nz = dz / length, -dx / length
    if (nx > 0) != (into > 0):
        nx, nz = -nx, -nz
    cx = 0.5 * (p0[0] + p1[0]) + 0.5 * thickness * nx
    cz = 0.5 * (p0[1] + p1[1]) + 0.5 * thickness * nz
    # rotation about y that turns the box's local +z into the face direction (dx, dz)
    pitch = math.degrees(math.atan2(dx, dz))
    return (cx, 0.0, cz), (0.5 * thickness, half_y, 0.5 * length + 0.002), pitch


def _box(x, y, z):
    """Axis-aligned box from (lo, hi) ranges -> (centre, half extents, 0 deg)."""
    return (
        (0.5 * (x[0] + x[1]), 0.5 * (y[0] + y[1]), 0.5 * (z[0] + z[1])),
        (0.5 * (x[1] - x[0]), 0.5 * (y[1] - y[0]), 0.5 * (z[1] - z[0])),
        0.0,
    )


def _mixer_collider_boxes():
    hx0, hx1 = CHUTE_HOLE_X
    hy = CHUTE_HOLE_HALF_Y
    lid_r = 0.122
    lid_z = (-0.011, 0.0)
    boxes = {
        # machine (measured from machine_rough): base plate, bowl seat, column, head, drive coupling
        "base_plate": _box((-0.296, 0.161), (-0.1115, 0.1115), (-0.233, -0.191)),
        "bowl_seat": _box((-0.090, 0.090), (-0.090, 0.090), (-0.191, -0.173)),
        "column": _box((-0.296, -0.125), (-0.1115, 0.1115), (-0.191, 0.052)),
        "head": _box((-0.296, 0.058), (-0.1115, 0.1115), (0.052, 0.106)),
        "coupling": _box((-0.032, 0.032), (-0.032, 0.032), (0.022, 0.052)),
        # bowl (inner radius 0.12 at the rim, bottom at z = -0.173), as a square box
        "bowl_bottom": _box((-0.12, 0.12), (-0.12, 0.12), (-0.183, -0.173)),
        "bowl_wall_px": _box((0.12, 0.13), (-0.13, 0.13), (-0.183, -0.011)),
        "bowl_wall_nx": _box((-0.13, -0.12), (-0.13, 0.13), (-0.183, -0.011)),
        "bowl_wall_py": _box((-0.12, 0.12), (0.12, 0.13), (-0.183, -0.011)),
        "bowl_wall_ny": _box((-0.12, 0.12), (-0.13, -0.12), (-0.183, -0.011)),
        # lid plate with the chute hole cut out (4 boxes around the hole)
        "lid_back": _box((-lid_r, hx0), (-lid_r, lid_r), lid_z),
        "lid_front": _box((hx1, lid_r), (-lid_r, lid_r), lid_z),
        "lid_side_py": _box((hx0, hx1), (hy, lid_r), lid_z),
        "lid_side_ny": _box((hx0, hx1), (-lid_r, -hy), lid_z),
        # raised parts of the lid: hub around the drive coupling, handle at the back
        "lid_hub": _box((-0.050, 0.050), (-0.063, 0.063), (0.0, 0.049)),
        "lid_handle": _box((-0.116, -0.045), (-0.053, 0.053), (0.0, 0.033)),
        # chute side walls (inner face at the hole's side)
        "chute_side_py": _box((0.058, 0.132), (hy, hy + 0.006), (0.0, 0.036)),
        "chute_side_ny": _box((0.058, 0.132), (-hy - 0.006, -hy), (0.0, 0.036)),
    }
    # chute back wall (solid wedge; its inner face leans back) and front lip (thin, leans forward)
    boxes["chute_back"] = _slanted_wall((0.0777, 0.0), (0.060, 0.049), 0.012, 0.058, into=-1)
    boxes["chute_front"] = _slanted_wall((0.1147, 0.0), (0.130, 0.0234), 0.006, 0.058, into=+1)
    return tuple((name, *box) for name, box in boxes.items())


MIXER_COLLIDER_BOXES = _mixer_collider_boxes()

# Bean regions in the mixer frame, as (lower, upper) corners of boxes [m].
MIXER_BOWL_REGION = ((-0.118, -0.118, -0.175), (0.118, 0.118, -0.012))
"""Below the lid inside the bowl: beans here went through the chute."""
MIXER_CHUTE_REGION = ((0.062, -0.052, -0.012), (0.128, 0.052, 0.024))
"""Inside the funnel, below its top opening, and in the hole through the lid plate."""
MIXER_BBOX_REGION = ((-0.300, -0.150, -0.235), (0.170, 0.150, 0.120))
"""The whole mixer; a resting bean in here that is not delivered lies on the lid / machine (= spilled)."""


##
# Resolved layout
##


@dataclass(frozen=True)
class KitchenLayout:
    """All poses the task needs, in the env (= U-layout world) frame."""

    table_top: float
    robot_pos: tuple[float, float, float]
    robot_yaw_deg: float
    robot_home: tuple[float, ...]
    gripper_open: float
    mixer_pos: tuple[float, float, float]
    """Position of the mixer frame (lid plate top), not of its bottom."""
    mixer_yaw_deg: float
    mixer_usd: str
    cup_pos: tuple[float, float, float]
    """Container base centre on the bottom table."""
    cup_yaw_deg: float
    source_table_edges: dict
    """Bottom table: x of its +x end, y of its front edge (for the real-world placement)."""

    @property
    def robot_rot(self) -> tuple[float, float, float, float]:
        return quat_xyzw_from_yaw(self.robot_yaw_deg)

    @property
    def mixer_rot(self) -> tuple[float, float, float, float]:
        return quat_xyzw_from_yaw(self.mixer_yaw_deg)

    @property
    def cup_rot(self) -> tuple[float, float, float, float]:
        return quat_xyzw_from_yaw(self.cup_yaw_deg)

    def mixer_point(self, local: tuple[float, float, float]) -> tuple[float, float, float]:
        """A point given in the mixer frame -> env frame (nominal mixer pose)."""
        x, y = u_layout().to_world(self.mixer_pos[0], self.mixer_pos[1], self.mixer_yaw_deg, local[0], local[1])
        return (x, y, self.mixer_pos[2] + local[2])

    @property
    def pour_target(self) -> tuple[float, float, float]:
        """Nominal point above the funnel that the container mouth is guided to (env frame)."""
        cx, cy, cz = CHUTE_TOP_CENTRE
        return self.mixer_point((cx, cy, cz + POUR_TARGET_ABOVE_CHUTE))


@lru_cache(maxsize=1)
def kitchen_layout() -> KitchenLayout:
    """Resolve the task poses from ``u_layout`` (scene defaults: no gap, side offset 0.5)."""
    lay = u_layout()
    places = lay.placements(lay.TABLE_LENGTH, lay.TABLE_WIDTH)
    rx, ry, rz, ryaw = lay.robot_pose(places, lay.TABLE_TOP)
    mx, my, myaw = lay.mixer_pose(places)
    # container: on the bottom table, measured from its +x end and its front edge (table frame: front = +y)
    px = lay.TABLE_LENGTH / 2.0 - CUP_CENTRE_FROM_END
    py = lay.TABLE_WIDTH / 2.0 - CUP_CENTRE_FROM_FRONT
    cx, cy, cyaw = lay.on_table(places, SOURCE_TABLE, px, py, CUP_YAW_DEG)
    end_x, front_y = lay.on_table(places, SOURCE_TABLE, lay.TABLE_LENGTH / 2.0, lay.TABLE_WIDTH / 2.0, 0.0)[:2]
    return KitchenLayout(
        table_top=lay.TABLE_TOP,
        robot_pos=(rx, ry, rz),
        robot_yaw_deg=ryaw,
        robot_home=tuple(lay.ROBOT_SCENE_POSE),
        gripper_open=lay.ROBOT_SCENE_GRIPPER,
        mixer_pos=(mx, my, lay.TABLE_TOP + lay.MIXER_BOTTOM_OFFSET),
        mixer_yaw_deg=myaw,
        mixer_usd=os.path.abspath(lay.MIXER_USD),
        cup_pos=(cx, cy, lay.TABLE_TOP),
        cup_yaw_deg=cyaw,
        source_table_edges={"end_x": end_x, "front_y": front_y},
    )


##
# Static colliders of the U layout (tables + Panda adapter), written as a small USD file
##

_ALU = (0.75, 0.77, 0.80)
_SHEET = (0.12, 0.13, 0.14)
_ADAPTER = (0.35, 0.35, 0.38)


def _table_boxes(length: float, width: float, top: float):
    """Boxes of one aluprofil table in its own frame (front = +y), following make_table.py's defaults: 80 mm
    profiles, 85 mm feet, 840 mm top flush with the front, 3 tall back uprights with crossbar and back sheet."""
    s, h = 0.08, 0.04
    plate, plate_w = 0.015, 0.84
    frame_top = top - plate
    back_top = 0.085 + 1.32  # feet + back uprights
    boxes = [
        ("top", (0.0, width / 2 - plate_w / 2, top - plate / 2), (length, plate_w, plate), _ALU),
        ("rail_front", (0.0, width / 2 - h, frame_top - h), (length, s, s), _ALU),
        ("rail_back", (0.0, -width / 2 + h, frame_top - h), (length, s, s), _ALU),
        ("crossbar", (0.0, -width / 2 + h, back_top + h), (length, s, s), _ALU),
        # back sheet on the inner face of the uprights, 420 mm high, top 50 mm above the crossbar
        ("back_sheet", (0.0, -width / 2 + s + 0.005, back_top + s + 0.05 - 0.21), (length, 0.01, 0.42), _SHEET),
    ]
    for k, x in enumerate((-length / 2 + h, 0.0, length / 2 - h)):
        boxes.append((f"leg_front_{k}", (x, width / 2 - h, (frame_top - s) / 2), (s, s, frame_top - s), _ALU))
        boxes.append((f"upright_{k}", (x, -width / 2 + h, back_top / 2), (s, s, back_top), _ALU))
    for k, x in enumerate((-length / 2 + h, length / 2 - h)):
        boxes.append((f"rail_side_{k}", (x, 0.0, frame_top - h), (s, width - 2 * s, s), _ALU))
    return boxes


def kitchen_statics_usda() -> str:
    """USD text of the static U layout: box colliders of the three tables and the Panda adapter plate."""
    lay = u_layout()
    places = lay.placements(lay.TABLE_LENGTH, lay.TABLE_WIDTH)
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Kitchen"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "Kitchen"',
        "{",
    ]

    def cube(name, centre, yaw, size, color):
        lines.extend(
            [
                f'    def Cube "{name}" (',
                '        prepend apiSchemas = ["PhysicsCollisionAPI"]',
                "    )",
                "    {",
                "        double size = 1",
                "        bool physics:collisionEnabled = 1",
                f"        color3f[] primvars:displayColor = [({color[0]}, {color[1]}, {color[2]})]",
                f"        double3 xformOp:translate = ({centre[0]:.6f}, {centre[1]:.6f}, {centre[2]:.6f})",
                f"        float xformOp:rotateZ = {yaw:.6f}",
                f"        float3 xformOp:scale = ({size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f})",
                '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ", "xformOp:scale"]',
                "    }",
            ]
        )

    for table, tx, ty, tyaw in places:
        for part, (px, py, pz), size, color in _table_boxes(lay.TABLE_LENGTH, lay.TABLE_WIDTH, lay.TABLE_TOP):
            wx, wy = lay.to_world(tx, ty, tyaw, px, py)
            cube(f"{table}_{part}", (wx, wy, pz), tyaw, size, color)

    ax, ay, _ = lay.adapter_pose(places)
    az = lay.TABLE_TOP + lay.ADAPTER_THICKNESS / 2
    lines.extend(
        [
            '    def Cylinder "PandaAdapter" (',
            '        prepend apiSchemas = ["PhysicsCollisionAPI"]',
            "    )",
            "    {",
            '        uniform token axis = "Z"',
            f"        double radius = {lay.ADAPTER_RADIUS}",
            f"        double height = {lay.ADAPTER_THICKNESS}",
            "        bool physics:collisionEnabled = 1",
            f"        color3f[] primvars:displayColor = [({_ADAPTER[0]}, {_ADAPTER[1]}, {_ADAPTER[2]})]",
            f"        double3 xformOp:translate = ({ax:.6f}, {ay:.6f}, {az:.6f})",
            '        uniform token[] xformOpOrder = ["xformOp:translate"]',
            "    }",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


_CONTAINER_COLOR = (0.92, 0.92, 0.88)


def container_usda(mass: float = 0.30) -> str:
    """USD text of the bean container: one rigid body (root prim) with visible box colliders.

    The boxes are the container's real shape up to the ~4 mm corner roundings and the rounded bottom of the handle
    leg; the full STL (1.6M triangles) is not needed without cameras. ``mass`` is authored on the root, PhysX
    derives centre of mass and inertia from the boxes (uniform material)."""
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Container"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "Container" (',
        '    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]',
        ")",
        "{",
        f"    float physics:mass = {float(mass):.6f}",
    ]
    for name, centre, half in container_collider_boxes():
        lines.extend(
            [
                f'    def Cube "{name}" (',
                '        prepend apiSchemas = ["PhysicsCollisionAPI"]',
                "    )",
                "    {",
                "        double size = 1",
                "        bool physics:collisionEnabled = 1",
                f"        color3f[] primvars:displayColor = [{_CONTAINER_COLOR}]",
                f"        double3 xformOp:translate = ({centre[0]:.6f}, {centre[1]:.6f}, {centre[2]:.6f})",
                f"        float3 xformOp:scale = ({2 * half[0]:.6f}, {2 * half[1]:.6f}, {2 * half[2]:.6f})",
                '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
                "    }",
            ]
        )
    lines.extend(["}", ""])
    return "\n".join(lines)


def _write_content_addressed(text: str, stem: str, directory: str | os.PathLike | None) -> str:
    """Write ``text`` once as ``<stem>_<hash>.usda`` (atomic) and return its path."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    folder = Path(directory) if directory is not None else Path(tempfile.gettempdir()) / "hestia_kitchen_pour"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}_{digest}.usda"
    if not path.is_file():
        tmp = folder / f".{path.name}.{os.getpid()}.tmp"
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    return str(path)


def write_kitchen_statics_usda(directory: str | os.PathLike | None = None) -> str:
    """Write :func:`kitchen_statics_usda` once (content-addressed, atomic) and return its path."""
    return _write_content_addressed(kitchen_statics_usda(), "kitchen_statics", directory)


def write_container_usda(directory: str | os.PathLike | None = None, mass: float = 0.30) -> str:
    """Write :func:`container_usda` once (content-addressed, atomic) and return its path."""
    return _write_content_addressed(container_usda(mass), "container01_v2", directory)


def container_footprint_world() -> dict[str, tuple[float, float, float, float]]:
    """Nominal footprint of the container on the bottom table, env frame: ``{part: (x_min, x_max, y_min, y_max)}``
    for the box (``"body"``) and the handle. Only valid for the axis-aligned placement (yaw 0 or 180 deg)."""
    lay = kitchen_layout()
    c, s = math.cos(math.radians(lay.cup_yaw_deg)), math.sin(math.radians(lay.cup_yaw_deg))
    if abs(s) > 1.0e-9:
        raise ValueError("container_footprint_world() needs an axis-aligned container (yaw 0 or 180 deg).")
    parts = {
        "body": (CONTAINER_BODY_X, (-CONTAINER_HALF_Y, CONTAINER_HALF_Y)),
        "handle": ((HANDLE_BAR_X[0], HANDLE_BAR_X[1]), (-HANDLE_HALF_Y, HANDLE_HALF_Y)),
    }
    out = {}
    for name, ((x0, x1), (y0, y1)) in parts.items():
        xs = sorted((lay.cup_pos[0] + c * x0, lay.cup_pos[0] + c * x1))
        ys = sorted((lay.cup_pos[1] + c * y0, lay.cup_pos[1] + c * y1))
        out[name] = (xs[0], xs[1], ys[0], ys[1])
    return out


def placement_report() -> str:
    """Where to put the real container on the middle table (for the real kitchen / sim-to-real)."""
    lay = kitchen_layout()
    end_x, front_y = lay.source_table_edges["end_x"], lay.source_table_edges["front_y"]
    fp = container_footprint_world()
    body, handle = fp["body"], fp["handle"]
    grasp = (lay.cup_pos[0] + HANDLE_GRASP_POINT[0], lay.cup_pos[1] + HANDLE_GRASP_POINT[1])
    return "\n".join(
        [
            "Container 01 v2 on the middle (bottom) table, sides parallel to the table edges, handle pointing to the "
            "table's end at the left table (robot side):",
            f"  footprint origin (centre of the 190 x 120 mm footprint incl. handle): "
            f"{100 * (end_x - lay.cup_pos[0]):.1f} cm from that end, {100 * (front_y - lay.cup_pos[1]):.1f} cm from "
            f"the front edge; env frame x {lay.cup_pos[0]:.4f} m, y {lay.cup_pos[1]:.4f} m, on the top at "
            f"z {lay.table_top:.3f} m",
            f"  handle tip -> table end: {100 * (end_x - handle[1]):.1f} cm; box wall at the handle -> table end: "
            f"{100 * (end_x - body[1]):.1f} cm; far box wall -> table end: {100 * (end_x - body[0]):.1f} cm",
            f"  front box wall -> front edge: {100 * (front_y - body[3]):.1f} cm; back box wall -> front edge: "
            f"{100 * (front_y - body[2]):.1f} cm; handle centre line -> front edge: "
            f"{100 * (front_y - lay.cup_pos[1]):.1f} cm",
            f"  handle grasp point (env frame): x {grasp[0]:.4f} m, y {grasp[1]:.4f} m, "
            f"z {lay.table_top + HANDLE_GRASP_POINT[2]:.4f} m",
        ]
    )
