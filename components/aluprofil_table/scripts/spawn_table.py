# HestiaSimulation (RoboTUM), author: Jeffrey

"""Spawn the aluminium-profile table USD in Isaac Lab (--drop-test also drops test cubes on it).

.. code-block:: bash

    # with a viewer
    uv run python ~/aluprofil_table/spawn_table.py --visualizer kit
    # headless check: drop test cubes, run 300 steps and print where they came to rest
    uv run python ~/aluprofil_table/spawn_table.py --drop-test --steps 300
    # a downloaded / converted table
    uv run python ~/aluprofil_table/spawn_table.py --usd ~/aluprofil_table/usd/downloaded_table.usd --visualizer kit
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import USD  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Spawn an aluminium-profile table.")
parser.add_argument(
    "--usd",
    type=str,
    default=os.path.join(USD, "alu_table.usd"),
    help="USD file of the table (create it with scripts/tools/convert_mesh.py).",
)
parser.add_argument("--table-height", type=float, default=0.68, help="Height of the table top [m] (for the cubes).")
parser.add_argument("--steps", type=int, default=0, help="Stop after this many steps (0 = run until closed).")
parser.add_argument("--drop-test", action="store_true", help="Drop 3 test cubes on the table top (check only).")
add_launcher_args(parser)
args_cli = parser.parse_args()

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg


def design_scene(usd_path: str, table_height: float, drop_test: bool = False) -> RigidObject | None:
    """Ground, light, the table (static collider) and, with drop_test, dynamic test cubes above it."""
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/defaultGroundPlane", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light.func("/World/Light", light)

    # The converted USD has a collider but no rigid body -> it is a static (fixed) table.
    table = sim_utils.UsdFileCfg(
        usd_path=usd_path,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.77, 0.8), metallic=0.8, roughness=0.35),
    )
    table.func("/World/AluTable", table, translation=(0.0, 0.0, 0.0))

    if not drop_test:
        return None
    # dynamic cubes dropped onto the table top
    for i, (x, y) in enumerate([(0.3, 0.2), (-0.3, -0.1), (0.0, 0.0)]):
        sim_utils.create_prim(f"/World/Drop{i}", "Xform", translation=(x, y, table_height + 0.3 + 0.1 * i))
    cubes = RigidObject(
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
    )
    return cubes


def main():
    usd_path = os.path.abspath(args_cli.usd)
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(f"{usd_path} not found - run the conversion step from ~/aluprofil_table/README.md")

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    with launch_simulation(sim_cfg, args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view([2.0, 1.8, 1.6], [0.0, 0.0, 0.5])
        cubes = design_scene(usd_path, args_cli.table_height, args_cli.drop_test)
        sim.reset()
        print("[INFO]: Setup complete...")
        count = 0
        while sim.is_running() and (args_cli.steps <= 0 or count < args_cli.steps):
            sim.step()
            count += 1
            if cubes is None:
                continue
            cubes.update(sim.get_physics_dt())
            if count % 100 == 0:
                print(f"[INFO] step {count}: cube z = {cubes.data.root_pos_w.torch[:, 2].tolist()}")


if __name__ == "__main__":
    main()
