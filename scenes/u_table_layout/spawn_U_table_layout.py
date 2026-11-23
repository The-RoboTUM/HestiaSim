"""Isaac Lab: the U table layout (u_layout.py) - three tables, the mixer, the Panda adapter and a Franka Panda.

The table USD is referenced three times and the mixer USD once (components/mixer, v2 by Mana); nothing
is copied. --drop-test adds 3 test cubes per table (check only).

The tables are of type no_brace (u_layout.TABLE_TYPE). Their USD is generated once per table design with
Isaac Lab's mesh converter, see components/aluprofil_table/scripts/README.md
(components/aluprofil_table/usd/alu_table_no_brace.usd).

.. code-block:: bash

    # from your IsaacLab folder
    uv run python <HestiaSimulation>/scenes/u_table_layout/spawn_U_table_layout.py --visualizer kit
    uv run python <HestiaSimulation>/scenes/u_table_layout/spawn_U_table_layout.py --drop-test --steps 300
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import u_layout  # noqa: E402

sys.path.insert(0, u_layout.TABLE_SCRIPTS)
from paths import USD  # noqa: E402  (the table component's paths)

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser(description="Spawn the U table layout (three tables + mixer).")
_tag = "" if u_layout.TABLE_TYPE == "standard" else f"_{u_layout.TABLE_TYPE}"
parser.add_argument("--table-usd", type=str, default=os.path.join(USD, f"alu_table{_tag}.usd"),
                    help="USD of ONE table (default: the scene's table type).")
parser.add_argument("--mixer-usd", type=str, default=u_layout.MIXER_USD, help="USD of the mixer.")
parser.add_argument("--no-mixer", action="store_true", help="Leave the mixer out.")
parser.add_argument("--adapter-usd", type=str, default=os.path.join(u_layout.ADAPTER_DIR, "usd", "panda_adapter.usd"),
                    help="USD of the Panda adapter plate (convert components/panda_adapter/meshes/panda_adapter.obj).")
parser.add_argument("--no-adapter", action="store_true", help="Leave the Panda adapter plate out.")
parser.add_argument("--no-robot", action="store_true", help="Leave the Franka Panda out.")
parser.add_argument("--gap", type=float, default=0.0, help="Gap between the bottom table and the side tables [m].")
parser.add_argument("--side-offset", type=float, default=0.5,
                    help="Side tables' back ends: 0 = in line with the bottom table's back edge, 0.5 = halfway.")
parser.add_argument("--table-top", type=float, default=u_layout.TABLE_TOP, help="Height of the table top [m].")
parser.add_argument("--steps", type=int, default=0, help="Stop after this many steps (0 = run until closed).")
parser.add_argument("--drop-test", action="store_true", help="Drop 3 test cubes on each table top (check only).")
add_launcher_args(parser)
args_cli = parser.parse_args()

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402

CUBE_XY = [(0.3, 0.2), (-0.3, -0.1), (0.0, 0.0)]  # per table, in the table's own frame


def design_scene(places, top: float):
    """Ground, light, three static tables, the mixer and (with --drop-test) test cubes."""
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/defaultGroundPlane", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light.func("/World/Light", light)

    # The converted table USD has a collider but no rigid body -> every copy is a static table.
    table = sim_utils.UsdFileCfg(
        usd_path=os.path.abspath(args_cli.table_usd),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.77, 0.8), metallic=0.8, roughness=0.35),
    )
    for name, x, y, yaw in places:
        table.func(f"/World/{name}", table, translation=(x, y, 0.0), orientation=u_layout.quat_xyzw(yaw))

    if not args_cli.no_mixer:
        # A mount Xform on the table top carries the table pose; the mixer keeps its own 0.233 m lift
        # under it (set explicitly, in case the spawner replaces the referenced root's transform).
        x, y, yaw = u_layout.mixer_pose(places)
        sim_utils.create_prim("/World/MixerMount", "Xform", translation=(x, y, top), orientation=u_layout.quat_xyzw(yaw))
        mixer = sim_utils.UsdFileCfg(usd_path=os.path.abspath(args_cli.mixer_usd))
        mixer.func("/World/MixerMount/Mixer", mixer, translation=(0.0, 0.0, u_layout.MIXER_BOTTOM_OFFSET))
        print(f"[INFO] Mixer        x={x:+.3f} y={y:+.3f} m, yaw={yaw:+.0f} deg, z={top:.3f} m")

    if not args_cli.no_adapter:
        # static plate (converted without mass) lying on the bottom table; robot faces ROBOT_FACING_DEG
        x, y, yaw = u_layout.adapter_pose(places)
        adapter = sim_utils.UsdFileCfg(usd_path=os.path.abspath(args_cli.adapter_usd))
        adapter.func("/World/PandaAdapter", adapter, translation=(x, y, top), orientation=u_layout.quat_xyzw(yaw))
        print(f"[INFO] Adapter      x={x:+.3f} y={y:+.3f} m, yaw={yaw:+.0f} deg, z={top:.3f} m")

    robot = None
    if not args_cli.no_robot and not args_cli.no_adapter:
        # components/franka_panda: the robot the pouring training uses (Isaac Lab's Panda + its settings)
        sys.path.insert(0, u_layout.FRANKA_DIR)
        import franka_cfg

        x, y, z, yaw = u_layout.robot_pose(places, top)
        robot_cfg = franka_cfg.isaac_cfg("/World/Robot")
        robot_cfg.init_state.pos = (x, y, z)
        robot_cfg.init_state.rot = u_layout.quat_xyzw(yaw)
        # scene start pose: the Panda's home pose (servo values / gravity stay from the component)
        robot_cfg.init_state.joint_pos.update({f"panda_joint{i}": q for i, q in enumerate(u_layout.ROBOT_SCENE_POSE, start=1)})
        robot_cfg.init_state.joint_pos["panda_finger_joint.*"] = u_layout.ROBOT_SCENE_GRIPPER
        robot = Articulation(robot_cfg)
        print(f"[INFO] Panda (components/franka_panda {franka_cfg.VERSION}) x={x:+.3f} y={y:+.3f} z={z:.3f} m, facing {yaw:+.0f} deg")

    if not args_cli.drop_test:
        return None, robot
    mx, my, _ = u_layout.mixer_pose(places)
    k = 0
    for name, x, y, yaw in places:
        for i, (cx, cy) in enumerate(CUBE_XY):
            wx, wy = u_layout.to_world(x, y, yaw, cx, cy)
            if not args_cli.no_mixer and (wx - mx) ** 2 + (wy - my) ** 2 < u_layout.MIXER_FOOTPRINT_RADIUS**2:
                continue  # this spot is under the mixer (solid in Isaac): no test cube there
            sim_utils.create_prim(f"/World/Drop{k}", "Xform", translation=(wx, wy, top + 0.3 + 0.1 * i))
            k += 1
    return RigidObject(
        RigidObjectCfg(
            prim_path="/World/Drop.*/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.08, 0.08, 0.08),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                mass_props=sim_utils.MassCfg(mass=0.5),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.3, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
        )
    ), robot


def main():
    needed = [args_cli.table_usd] + ([] if args_cli.no_mixer else [args_cli.mixer_usd])
    needed += [] if args_cli.no_adapter else [args_cli.adapter_usd]
    for path in needed:
        if not os.path.isfile(os.path.abspath(path)):
            raise FileNotFoundError(f"{path} not found (table USD: convert it first, see the table's scripts/README.md)")
    places = u_layout.placements(u_layout.TABLE_LENGTH, u_layout.TABLE_WIDTH, args_cli.gap, args_cli.side_offset)
    for name, x, y, yaw in places:
        print(f"[INFO] {name:12s} x={x:+.3f} y={y:+.3f} m, yaw={yaw:+.0f} deg")

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view([0.0, -4.5, 3.5], [0.0, 0.6, 0.5])
        cubes, robot = design_scene(places, args_cli.table_top)
        sim.reset()
        print("[INFO]: Setup complete...")
        count = 0
        while sim.is_running() and (args_cli.steps <= 0 or count < args_cli.steps):
            if robot is not None:  # hold the robot in its default (home) joint pose
                robot.set_joint_position_target(robot.data.default_joint_pos)
                robot.write_data_to_sim()
            sim.step()
            count += 1
            if robot is not None:
                robot.update(sim.get_physics_dt())
            if cubes is None:
                continue
            cubes.update(sim.get_physics_dt())
            if count % 100 == 0:
                print(f"[INFO] step {count}: cube z = {[round(z, 3) for z in cubes.data.root_pos_w.torch[:, 2].tolist()]}")


if __name__ == "__main__":
    main()
