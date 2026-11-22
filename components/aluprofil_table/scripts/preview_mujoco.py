"""Laptop preview of the table in MuJoCo (no NVIDIA GPU / Isaac Lab needed).

Shows the static table. With --drop-test, three 8 cm, 0.5 kg test cubes are dropped on the top
(as in spawn_table.py) to check that the top is solid; without it there are no cubes.
Takes the same flags as make_table.py, so you can iterate on the design here and only run the
USD conversion + spawn_table.py on the Isaac machine once you're happy.

.. code-block:: bash

    cd ~/Projects/RoboTUM/aluprofil_table
    UV="uv run --no-project --with mujoco --with trimesh --with shapely --with mapbox-earcut --with scipy --with networkx"
    $UV python preview_mujoco.py                       # viewer, default table
    $UV python preview_mujoco.py --drop-test --steps 300   # headless check: print where test cubes settle
    $UV python preview_mujoco.py --leg 0.60 --plank-along y
    $UV python preview_mujoco.py --obj meshes/downloaded_table.obj --table-height 0.68
    # three tables in a U: see u_layout/preview_U_table_layout.py

Collision caveat: MuJoCo collides every mesh as its CONVEX HULL. Each generated part is
convex-ish (a profile's hull is its square box, a plank's hull is a slab), so one geom per part is
accurate enough for contacts; T-slot grooves are visual only. For --obj, the mesh is split into
connected pieces and each piece gets its own hull - a single merged blob would fill in the space
under the top. Isaac Lab's triangleMesh collider (static table) has no such limitation.
"""

import os
import sys
import time
import warnings

import mujoco
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_table import build_from_args, build_parser  # noqa: E402

def quat_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    """Rotation about z as a MuJoCo quaternion (w, x, y, z)."""
    h = np.radians(yaw_deg) / 2
    return (float(np.cos(h)), 0.0, 0.0, float(np.sin(h)))


def to_world(x: float, y: float, yaw_deg: float, px: float, py: float) -> tuple[float, float]:
    """Point (px, py) in a table's own frame -> world, for a table placed at (x, y, yaw)."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    return float(x + c * px - s * py), float(y + s * px + c * py)

CUBE_XY = [(0.3, 0.2), (-0.3, -0.1), (0.0, 0.0)]  # same as spawn_table.py
CUBE_SIZE, CUBE_MASS = 0.08, 0.5


def load_parts(args) -> tuple[dict[str, trimesh.Trimesh], float]:
    """Return (name -> mesh) and the table-top height."""
    if args.obj:
        mesh = trimesh.load(args.obj, force="mesh")
        pieces = mesh.split(only_watertight=False)
        print(f"[INFO] {args.obj}: {len(pieces)} connected pieces -> one convex-hull geom each")
        return {f"piece_{i}": p for i, p in enumerate(pieces)}, args.table_height
    parts = build_from_args(args)
    top = parts["plate_plank_0"].bounds[1][2] if "plate_plank_0" in parts else max(m.bounds[1][2] for m in parts.values())
    return parts, top


def build_model(parts: dict[str, trimesh.Trimesh], top: float, placements=(('Table', 0.0, 0.0, 0.0),),
                drop_test: bool = False, extras=(), attachments=()) -> mujoco.MjModel:
    """extras: other static objects for a scene, (name, mesh in meters, (x, y, z), yaw_deg, rgba), visual only.
    attachments: other MuJoCo models (e.g. a robot) merged into the scene, dicts with
    file (MJCF path), prefix (name prefix), pos (x, y, z), yaw (deg) and optionally
    key (keyframe name for the start pose), joint_targets ({joint: value}: start pose + actuator targets),
    gains ({actuator: (stiffness, damping, force_limit)}), gravcomp (True: no gravity on that model)."""
    assets, mesh_xml, geom_xml = {}, [], []
    split = {}
    for name, m in parts.items():  # one convex hull per connected piece (e.g. notched planks)
        pieces = m.split(only_watertight=False)
        for i, piece in enumerate(pieces):
            split[name if len(pieces) == 1 else f"{name}_{i}"] = piece
    for name, m in split.items():
        if len(m.faces) < 4:
            continue
        fname = f"{name}.stl"
        assets[fname] = m.export(file_type="stl")
        mesh_xml.append(f'<mesh name="{name}" file="{fname}"/>')
        geom_xml.append(name)
    # one static body per table placement, all sharing the same meshes; 3 test cubes per table only with drop_test
    tables, cube_bodies, k = [], [], 0
    for tname, tx, ty, yaw in placements:
        q = " ".join(f"{v:.6f}" for v in quat_wxyz(yaw))
        geoms = "".join(f'<geom name="{tname}_{g}" type="mesh" mesh="{g}" material="alu"/>' for g in geom_xml)
        tables.append(f'<body name="{tname}" pos="{tx} {ty} 0" quat="{q}">{geoms}</body>')
        for i, (cx, cy) in enumerate(CUBE_XY if drop_test else []):
            wx, wy = to_world(tx, ty, yaw, cx, cy)
            cube_bodies.append(
                f'''<body name="cube{k}" pos="{wx:.4f} {wy:.4f} {top + 0.3 + 0.1 * i}">
              <freejoint/>
              <geom type="box" size="{CUBE_SIZE / 2} {CUBE_SIZE / 2} {CUBE_SIZE / 2}" mass="{CUBE_MASS}" rgba="0.9 0.3 0.1 1"/>
            </body>'''
            )
            k += 1
    cubes = "\n".join(cube_bodies)
    for ename, emesh, (ex, ey, ez), eyaw, rgba in extras:  # visual only: no collision (contype/conaffinity 0)
        fname = f"extra_{ename}.stl"
        assets[fname] = emesh.export(file_type="stl")
        mesh_xml.append(f'<mesh name="extra_{ename}" file="{fname}"/>')
        q = " ".join(f"{v:.6f}" for v in quat_wxyz(eyaw))
        tables.append(f'<body name="{ename}" pos="{ex} {ey} {ez}" quat="{q}"><geom type="mesh" mesh="extra_{ename}" '
                      f'rgba="{" ".join(map(str, rgba))}" contype="0" conaffinity="0"/></body>')
    xml = f"""
    <mujoco model="alu_table_preview">
      <option timestep="0.002" integrator="implicitfast"/>
      <visual><global azimuth="135" elevation="-25"/></visual>
      <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".25 .26 .28" rgb2=".3 .31 .33" width="512" height="512"/>
        <material name="floor" texture="grid" texrepeat="8 8"/>
        <material name="alu" rgba="0.75 0.77 0.8 1" specular="0.6" shininess="0.6"/>
        {''.join(mesh_xml)}
      </asset>
      <worldbody>
        <light pos="0 0 4" dir="0 0 -1" directional="true"/>
        <geom name="floor" type="plane" size="5 5 0.1" material="floor"/>
        {''.join(tables)}
        {cubes}
      </worldbody>
    </mujoco>"""
    if not attachments:
        return mujoco.MjModel.from_xml_string(xml, assets)
    spec = mujoco.MjSpec.from_string(xml, assets=assets)
    for att in attachments:
        child = mujoco.MjSpec.from_file(att["file"])
        if att.get("gravcomp"):  # like Isaac Lab's disable_gravity: gravity compensated on every body of the model
            for body in child.bodies:
                body.gravcomp = 1.0
        frame = spec.worldbody.add_frame(pos=list(att["pos"]), quat=list(quat_wxyz(att["yaw"])))
        with warnings.catch_warnings():  # the attached model loads its own meshes; MuJoCo's asset-dict note is harmless
            warnings.simplefilter("ignore")
            spec.attach(child, frame=frame, prefix=att["prefix"])
    return spec.compile()


def apply_start_poses(model, data, attachments):
    """Set each attached model's joints and actuator targets from its keyframe (only its own joints, so
    other free bodies such as test cubes keep their positions)."""
    for att in attachments:
        p = att["prefix"]
        for name, (kp, kd, force) in att.get("gains", {}).items():  # position servo: f = kp*(target-q) - kd*qdot
            a = model.actuator(p + name).id
            model.actuator_gainprm[a, 0] = kp
            model.actuator_biasprm[a, :3] = (0.0, -kp, -kd)
            model.actuator_forcerange[a] = (-force, force)
            model.actuator_forcelimited[a] = 1
        for name, value in att.get("joint_targets", {}).items():
            j = model.joint(p + name).id
            data.qpos[model.jnt_qposadr[j]] = value
            for a in range(model.nu):  # actuator driving this joint directly gets the same target
                if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[a, 0] == j:
                    data.ctrl[a] = value
        if not att.get("key"):
            continue
        key = model.key(p + att["key"])
        for j in range(model.njnt):
            if model.joint(j).name.startswith(att["prefix"]):
                adr = model.jnt_qposadr[j]
                data.qpos[adr] = key.qpos[adr]
        for a in range(model.nu):
            if model.actuator(a).name.startswith(att["prefix"]):
                data.ctrl[a] = key.ctrl[a]
    mujoco.mj_forward(model, data)


def main():
    parser = build_parser()
    parser.description = "Preview the aluminium-profile table in MuJoCo."
    parser.add_argument("--obj", type=str, default="", help="Preview this mesh instead of generating the table.")
    parser.add_argument("--table-height", type=float, default=0.68, help="Top height [m] for --obj (cube start).")
    parser.add_argument("--steps", type=int, default=0, help="Headless: simulate this many 10 ms steps, print, exit.")
    parser.add_argument("--drop-test", action="store_true", help="Drop 3 test cubes on each table top (check only).")
    args = parser.parse_args()

    parts, top = load_parts(args)
    run(parts, top, [("Table", 0.0, 0.0, 0.0)], args.steps, args.drop_test)


def run(parts, top: float, places, steps: int = 0, drop_test: bool = False, extras=(), camera=None, attachments=()):
    """camera: optional (azimuth_deg, elevation_deg, distance_m) for the viewer's start view."""
    """Build the scene for the given table placements (name, x, y, yaw_deg) and run it."""
    model = build_model(parts, top, places, drop_test, extras, attachments)
    data = mujoco.MjData(model)
    apply_start_poses(model, data, attachments)
    n_cubes = len(CUBE_XY) * len(places) if drop_test else 0
    print(f"[INFO] {len(places)} table(s) x {len(parts)} parts -> {model.ngeom - 1 - n_cubes} table geoms, "
          f"table top z = {top:.4f} m")
    cube_ids = [model.body(f"cube{i}").id for i in range(n_cubes)]

    if steps > 0:
        substeps = round(0.01 / model.opt.timestep)
        for _ in range(steps):
            mujoco.mj_step(model, data, nstep=substeps)
        if n_cubes:
            z = data.xpos[cube_ids, 2]
            print(f"[INFO] after {steps * 0.01:.1f} s: cube z = {np.round(z, 4).tolist()}")
            print(f"[INFO] expected on the top: {top + CUBE_SIZE / 2:.4f} m")
        else:
            print(f"[INFO] simulated {steps * 0.01:.1f} s (no test cubes; add --drop-test to check the top)")
        return

    from mujoco import viewer as mj_viewer

    with mj_viewer.launch_passive(model, data) as viewer:
        xs = [p[1] for p in places]
        ys = [p[2] for p in places]
        viewer.cam.lookat[:] = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, 0.5)
        viewer.cam.distance = 3.2 if len(places) == 1 else 6.0
        if camera is not None:
            viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = camera
        while viewer.is_running():
            t0 = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - t0)))


if __name__ == "__main__":
    main()
