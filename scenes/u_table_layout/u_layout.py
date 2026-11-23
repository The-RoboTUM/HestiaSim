"""U table layout: three aluprofil tables (type no_brace) in a U, with the mixer on the left table.

Pure Python (no trimesh / Isaac imports), shared by the MuJoCo preview and the Isaac Lab spawn script.
This scene only REFERENCES components (components/aluprofil_table, components/mixer); it never copies
their geometry.

Table frame (as built by make_table.py): x = along the length, y = depth with the FRONT at +y and
the back wall at -y, origin on the floor at the table centre. A placement is (name, x, y, yaw):
position of that table's origin in meters and its rotation about the vertical axis in degrees.

LEFT / RIGHT are as seen standing in the opening of the U, looking at the bottom table.
Seen from above (bottom table at the back, its front = +y; the left table is at +x):

                  north (-y)
           ┌──────────────────┐
     ┌────┐│ C   TableBottom  │┌────┐      back walls on the outside, fronts facing the inside;
     │ A  │└──────────────────┘│    │      side tables OUTSIDE the ends of the bottom table;
     │ L  │ M─►    inside      │ R  │      their back ends start halfway along the bottom
     │    │                    │    │      table's depth (side_offset = 0.5)
     └────┘                    └────┘      M = mixer in the middle of the left table, perpendicular
     west (+x)  ▲ you stand here  east (-x)    to it, its front (bowl side) facing the inside of the U
     (Correction 2026-11-23: this sketch used to show L and R mirrored; the numbers were always right.)
     A = Panda adapter plate on the left table at its end next to the bottom table, robot facing
         45 deg "south-east": towards the inside and the opening of the U (see ADAPTER_* below)
     C = container on the bottom table at its end next to the robot, handle pointing west (CONTAINER_* below)

gap         = space between the bottom table's ends and the side tables (default 0 = touching)
side_offset = where the side tables' back ends start, as a fraction of the bottom table's depth
              from its back edge: 0 = in line with the back edge, 0.5 = halfway (default), 1 = front edge
With three 1800 x 960 tables and side_offset 0.5: overall 3720 x 2280 mm, inside opening 1800 wide.
"""

import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TABLE_SCRIPTS = os.path.join(REPO, "components", "aluprofil_table", "scripts")
MIXER_DIR = os.path.join(REPO, "components", "mixer")
ADAPTER_DIR = os.path.join(REPO, "components", "panda_adapter")
ADAPTER_SCRIPTS = os.path.join(ADAPTER_DIR, "scripts")

TABLE_TYPE = "no_brace"  # make_table.py --type: the tables in this scene have no diagonal braces
TABLE_LENGTH = 1.80  # m, make_table.py --length
TABLE_WIDTH = 0.96  # m, make_table.py --width
TABLE_TOP = 0.68  # m, top surface height of the table (make_table.py defaults)

# Mixer (components/mixer, v2 = Mana's model, the current one, confirmed 2026-11-22). Its USD lifts its own root by
# 0.233 m so the mixer's bottom is at z = 0; footprint x -0.296..+0.161, y -0.147..+0.147 m, height 0.339 m
# (in its own frame). Its FRONT is +x: the bowl side (the body/column extends to -x, the chute is at +x).
MIXER_USD = os.path.join(MIXER_DIR, "mum8_chute_v2.usda")
MIXER_STL = os.path.join(MIXER_DIR, "mum8_chute_v2.stl")  # same shape, mm
MIXER_BOTTOM_OFFSET = 0.233  # m, the USD root's own translate (bottom of the mixer -> z = 0)
# where it stands: (table name, x, y on that table's top in the TABLE frame [m]). It stands perpendicular to
# the table with its front facing the table's front (= the inside of the U). x = 0: middle of the table's
# length; y = 0.1275: its footprint (-0.296..+0.161 along its front direction) centred on the top's depth.
MIXER_ON = ("TableLeft", 0.0, 0.1275)
MIXER_FOOTPRINT_RADIUS = 0.33  # m, covers the footprint from its origin (0.296 back, 0.147 sideways)


# Panda adapter plate (components/panda_adapter): Ø350 x 10 mm, origin = plate centre on its bottom face,
# +y = robot back (connector end). It lies on the LEFT table at its end next to the bottom table (the
# corner of the U), its outer edge 75 mm from that end and 100 mm from the table's front edge (the edge
# facing the inside of the U). The robot faces 45 deg diagonally "south-east": halfway between the inside
# of the U (-x) and the opening of the U (+y), i.e. world yaw +135 deg.
# Compass as in the viewer's start view (standing in the opening, looking at the bottom table):
# north = towards the bottom table (-y), south = towards the opening (+y), east = -x, west = +x.
# (Correction 2026-11-22: first faced north-east, -135 deg, into the corner; it should face south-east.)
# (Correction 2026-11-22: first placed on the bottom table; it belongs on the left table.)
ADAPTER_TABLE = "TableLeft"
ADAPTER_END = -1  # which end of that table, in its own frame: -1 = local -x (for TableLeft: next to the bottom table)
ADAPTER_RADIUS = 0.175  # m
ADAPTER_THICKNESS = 0.010  # m
ADAPTER_EDGE_TO_END = 0.075  # m, plate edge -> the table's end (earlier tries: 200, 150 mm)
ADAPTER_EDGE_TO_FRONT = 0.100  # m, plate edge -> bottom table's front edge
ROBOT_FACING_DEG = 135.0  # world yaw of the robot's front ("south-east", see the compass above)
JOINT1_IN_PLATE = (0.0, 0.0085)  # m, robot's vertical axis in plate coordinates (8.5 mm off-centre, measured)


# Container (components/container, container01 v3 = simplified copy of v2, 50k triangles): 190 x 120 x 70 mm, STL in mm, origin = centre of its bottom
# face. Body x -0.095..+0.065 m, the HANDLE sticks out at +x (+0.065..+0.095 m, 20 mm wide) in its own frame.
# It stands on the BOTTOM table (the one in front of you in the opening) at its end next to the robot, the
# handle pointing west (+x world, towards the left table / the robot), square to the table (yaw 0) for now;
# later the handle should point at the robot itself. Centre 15 cm from the bottom table's +x end and 20 cm from
# its front edge = the same spot as the kitchen pour training (training/.../kitchen_pour/layout.py,
# CUP_CENTRE_FROM_END / CUP_CENTRE_FROM_FRONT); keep the two equal. Fixed spot for now: the container will be
# spawned at random positions later. (2026-11-23: moved from handle tip 150 mm / body 100 mm from the edges.)
CONTAINER_DIR = os.path.join(REPO, "components", "container")
CONTAINER_STL = os.path.join(CONTAINER_DIR, "container01_v3.stl")  # mm, 2.5 MB (v2 = 79 MB full detail)
CONTAINER_TABLE = "TableBottom"
CONTAINER_CENTRE_FROM_END = 0.15  # m, container origin -> the bottom table's +x end (= kitchen pour)
CONTAINER_CENTRE_FROM_FRONT = 0.20  # m, container origin -> the bottom table's front edge, +y (= kitchen pour)
CONTAINER_YAW = 0.0  # deg relative to the table: 0 = handle along the table's +x = west


def container_pose(places, length: float = TABLE_LENGTH, width: float = TABLE_WIDTH):
    """World (x, y, yaw) of the container (origin = bottom centre) on CONTAINER_TABLE's top."""
    px = length / 2 - CONTAINER_CENTRE_FROM_END
    py = width / 2 - CONTAINER_CENTRE_FROM_FRONT
    return on_table(places, CONTAINER_TABLE, px, py, CONTAINER_YAW)


def adapter_pose(places, length: float = TABLE_LENGTH, width: float = TABLE_WIDTH):
    """World (x, y, yaw) of the adapter plate centre. Plate yaw = robot facing + 90 deg, because the
    robot's front is the plate's -y direction."""
    px = ADAPTER_END * (length / 2 - ADAPTER_EDGE_TO_END - ADAPTER_RADIUS)
    py = width / 2 - ADAPTER_EDGE_TO_FRONT - ADAPTER_RADIUS  # the top is flush with the table's front edge (+y)
    x, y, _ = on_table(places, ADAPTER_TABLE, px, py, 0.0)
    return x, y, ROBOT_FACING_DEG + 90.0


# Franka Panda on the adapter: components/franka_panda (franka_cfg.py) = the robot the pouring training
# uses (settings v1, not final). Isaac Lab: franka_cfg.isaac_cfg(). MuJoCo preview: the same settings
# (franka_cfg.mujoco_settings()) applied to MuJoCo Menagerie franka_emika_panda/panda.xml (same Franka
# description, links and joints; not part of this repo: set HESTIA_PANDA_MJCF or pass --panda-xml).
# (2026-11-22: replaces the earlier choice of Isaac Lab's default / high_pd Panda, which did not match the training.)
# Base (link0) origin = the robot's vertical axis (joint 1) on the base's bottom face, +x = robot front.
FRANKA_DIR = os.path.join(REPO, "components", "franka_panda")
# The scene spawns the robot in the Panda's upright "home" pose (MuJoCo Menagerie keyframe "home"), not in the
# pouring training's start pose (that one is folded low near the table and only makes sense with its cups).
# Servo values and gravity still come from components/franka_panda.
ROBOT_SCENE_POSE = (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853)  # joints 1-7 [rad]
ROBOT_SCENE_GRIPPER = 0.04  # m per finger (open)
PANDA_MJCF = os.environ.get(
    "HESTIA_PANDA_MJCF",
    os.path.expanduser("~/Projects/RoboTUM/franka-sandbox/third_party/mujoco_menagerie/franka_emika_panda/panda.xml"),
)


def robot_pose(places, top: float = TABLE_TOP):
    """World (x, y, z, yaw) of the robot base: on the adapter's top surface, joint 1 at its measured spot."""
    x, y = joint1_world(places)
    return x, y, top + ADAPTER_THICKNESS, ROBOT_FACING_DEG


def joint1_world(places):
    """World (x, y) of the robot's vertical axis (where a robot base would be placed)."""
    x, y, yaw = adapter_pose(places)
    return to_world(x, y, yaw, *JOINT1_IN_PLATE)


def placements(length: float = TABLE_LENGTH, width: float = TABLE_WIDTH, gap: float = 0.0, side_offset: float = 0.5):
    side_x = length / 2 + gap + width / 2  # side table centre: outside the bottom table's ends
    side_y = -width / 2 + side_offset * width + length / 2  # its back end at side_offset of the depth
    return [
        ("TableBottom", 0.0, 0.0, 0.0),
        ("TableLeft", side_x, side_y, 90.0),  # at +x; its front (+y) turned to -x: faces the inside
        ("TableRight", -side_x, side_y, -90.0),  # at -x; its front (+y) turned to +x: faces the inside
    ]


def mixer_pose(places):
    """World (x, y, yaw) of the mixer: on MIXER_ON, perpendicular to the table, its front (+x) turned to the
    table's front (+y of the table frame = the inside of the U), i.e. yaw = table yaw + 90 deg."""
    return on_table(places, *MIXER_ON, 90.0)


def on_table(places, table: str, px: float, py: float, yaw_rel: float):
    """World (x, y, yaw) of an object standing at (px, py, yaw_rel) on the given table's top."""
    _, tx, ty, tyaw = next(p for p in places if p[0] == table)
    x, y = to_world(tx, ty, tyaw, px, py)
    return x, y, tyaw + yaw_rel


def to_world(x: float, y: float, yaw_deg: float, px: float, py: float) -> tuple[float, float]:
    """Point (px, py) in a table's own frame -> world, for a table placed at (x, y, yaw)."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return x + c * px - s * py, y + s * px + c * py


def quat_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    """Rotation about z as a quaternion (w, x, y, z), the convention of MuJoCo (and of Isaac Lab before 3.0)."""
    h = math.radians(yaw_deg) / 2
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def quat_xyzw(yaw_deg: float) -> tuple[float, float, float, float]:
    """Rotation about z as a quaternion (x, y, z, w), the convention of Isaac Lab 3.x (create_prim, spawners,
    InitialStateCfg.rot). (Fix 2026-11-23: the Isaac spawn script used quat_wxyz, which Isaac Lab 3 reads as a
    180 deg roll for yaw 0, i.e. upside-down tables.)"""
    h = math.radians(yaw_deg) / 2
    return (0.0, 0.0, math.sin(h), math.cos(h))
