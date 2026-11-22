"""Laptop preview of the U table layout in MuJoCo: three tables + the mixer on the left table.

The mixer is shown for placement only (visual, no collision). --drop-test adds 3 test cubes per table.

.. code-block:: bash

    cd <HestiaSimulation>/scenes/u_table_layout
    UV="uv run --no-project --with mujoco --with trimesh --with shapely --with mapbox-earcut --with scipy --with networkx"
    $UV python preview_U_table_layout.py                           # viewer
    $UV python preview_U_table_layout.py --drop-test --steps 300   # check: test cubes rest at top + 0.04 m
    $UV python preview_U_table_layout.py --no-mixer                # tables only
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import u_layout  # noqa: E402

sys.path.insert(0, u_layout.TABLE_SCRIPTS)  # the table component's scripts
import preview_mujoco  # noqa: E402
import trimesh  # noqa: E402
from make_table import build_parser  # noqa: E402


def mixer_extra(places, top: float):
    mesh = trimesh.load(u_layout.MIXER_STL, force="mesh")
    mesh.apply_scale(0.001)  # STL in mm -> m
    mesh.apply_translation((0, 0, u_layout.MIXER_BOTTOM_OFFSET))  # bottom of the mixer -> z = 0 (as its USD)
    x, y, yaw = u_layout.mixer_pose(places)
    print(f"[INFO] Mixer        x={x:+.3f} y={y:+.3f} m, yaw={yaw:+.0f} deg, on {u_layout.MIXER_ON[0]} at z={top:.3f}")
    return ("Mixer", mesh, (x, y, top), yaw, (0.85, 0.85, 0.88, 1.0))


def adapter_extra(places, top: float):
    sys.path.insert(0, u_layout.ADAPTER_SCRIPTS)
    import make_adapter  # the adapter component's generator (only build_plate() is used here)

    plate = make_adapter.build_plate(make_adapter.build_parser().parse_args([]))  # metres, origin bottom centre
    x, y, yaw = u_layout.adapter_pose(places)
    jx, jy = u_layout.joint1_world(places)
    print(f"[INFO] Adapter      x={x:+.3f} y={y:+.3f} m, yaw={yaw:+.0f} deg (robot faces {u_layout.ROBOT_FACING_DEG:.0f} deg), "
          f"robot axis at ({jx:+.3f}, {jy:+.3f}), z={top + u_layout.ADAPTER_THICKNESS:.3f}")
    return ("Adapter", plate, (x, y, top), yaw, (0.62, 0.64, 0.68, 1.0))


def robot_attachment(places, top: float, panda_xml: str, robot_cfg: str = "default"):
    if not os.path.isfile(panda_xml):
        print(f"[WARN] Panda model not found: {panda_xml}\n       Get MuJoCo Menagerie (franka_emika_panda) and pass --panda-xml "
              "or set HESTIA_PANDA_MJCF. Showing the scene without the robot.")
        return []
    x, y, z, yaw = u_layout.robot_pose(places, top)
    att = dict(file=panda_xml, prefix="panda/", pos=(x, y, z), yaw=yaw)
    if robot_cfg == "menagerie":  # Menagerie's own pose and servo values
        att["key"] = "home"
        how = "Menagerie 'home' pose and gains"
    else:  # Isaac Lab's demo Panda: its start pose and arm servo values
        kp, kd, gravity = u_layout.ISAAC_PANDA_ARM[robot_cfg]
        att["joint_targets"] = dict(u_layout.ISAAC_PANDA_START)
        att["gains"] = {f"actuator{i}": (kp, kd, f) for i, f in u_layout.ISAAC_PANDA_FORCE.items()}
        att["gravcomp"] = not gravity
        how = f"Isaac Lab '{robot_cfg}': start pose, stiffness {kp:g}, damping {kd:g}, gravity {'on' if gravity else 'off'}"
    print(f"[INFO] Panda        x={x:+.3f} y={y:+.3f} z={z:.3f} m, facing {yaw:+.0f} deg ({how})")
    return [att]


def main():
    parser = build_parser()
    parser.description = "Preview the U table layout (three tables + mixer) in MuJoCo."
    parser.set_defaults(type=u_layout.TABLE_TYPE)  # the scene's table type (override with --type)
    parser.add_argument("--gap", type=float, default=0.0, help="Gap between the bottom table and the side tables [m].")
    parser.add_argument("--side-offset", type=float, default=0.5,
                        help="Side tables' back ends: 0 = in line with the bottom table's back edge, 0.5 = halfway.")
    parser.add_argument("--no-mixer", action="store_true", help="Leave the mixer out.")
    parser.add_argument("--no-adapter", action="store_true", help="Leave the Panda adapter plate out.")
    parser.add_argument("--no-robot", action="store_true", help="Leave the Franka Panda out.")
    parser.add_argument("--panda-xml", type=str, default=u_layout.PANDA_MJCF, help="MuJoCo Menagerie panda.xml.")
    parser.add_argument("--robot-cfg", choices=("default", "high_pd", "menagerie"), default="default",
                        help="Panda settings: Isaac Lab default / high_pd (as the Isaac script), or Menagerie's own.")
    parser.add_argument("--steps", type=int, default=0, help="Headless: simulate this many 10 ms steps, print, exit.")
    parser.add_argument("--drop-test", action="store_true", help="Drop 3 test cubes on each table top (check only).")
    # load_parts() also reads these; not meaningful here, so hidden
    parser.add_argument("--obj", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--table-height", type=float, default=0.68, help=argparse.SUPPRESS)
    args = parser.parse_args()
    parts, top = preview_mujoco.load_parts(args)
    places = u_layout.placements(args.length, args.width, args.gap, args.side_offset)
    for name, x, y, yaw in places:
        print(f"[INFO] {name:12s} x={x:+.3f} y={y:+.3f} m, yaw={yaw:+.0f} deg")
    extras = ([] if args.no_mixer else [mixer_extra(places, top)]) + ([] if args.no_adapter else [adapter_extra(places, top)])
    # start view: standing in the opening of the U, looking at the bottom table (left table on the left)
    robot = [] if (args.no_robot or args.no_adapter) else robot_attachment(places, top, args.panda_xml, args.robot_cfg)
    preview_mujoco.run(parts, top, places, args.steps, args.drop_test, extras, camera=(-90.0, -30.0, 6.5),
                       attachments=robot)


if __name__ == "__main__":
    main()
