# HestiaSim (RoboTUM), author: Jeffrey

"""Franka Panda as used by the Hestia pouring training - one place for the robot's settings.

Version v1 = the values of training/hestia_beans_pour/src/hestia_beans_pour/tasks/pour/pour_env_cfg.py
as of 2026-11-22 (task HestiaBeansPour-Franka-Pour, Isaac Lab dev checkout, isaaclab 47.0.0). The values
are NOT final yet; when the training changes them, update this file (and bump VERSION), ideally by having
the training import isaac_cfg() from here.

The robot files themselves come with Isaac Lab (isaaclab_assets FRANKA_PANDA_CFG and its USD); nothing is
copied here. Robot base frame: link0 origin = joint 1 axis on the base's bottom face, +x = robot front.

This module imports nothing from Isaac Lab at the top, so the MuJoCo preview can read the values without it.
"""

VERSION = "v1"
SOURCE = "training/hestia_beans_pour/.../tasks/pour/pour_env_cfg.py, 2026-11-22"

# Isaac Lab robot asset (relative to ISAACLAB_NUCLEUS_DIR) and its USD variants
USD_ASSET_PATH = "Robots/FrankaEmika/franka_panda.usda"
USD_VARIANTS = {"Physics": "physx", "Colliders": "convex_hulls"}
GRAVITY_ON_ROBOT = False  # gravity disabled on the robot links (replaces MuJoCo gravity compensation)

# arm joint servo (Isaac Lab ImplicitActuator "panda_arm"), keys = Isaac Lab joint-name patterns
ARM_STIFFNESS = {"panda_joint[1-2]": 1000.0, "panda_joint[3-4]": 750.0, "panda_joint[5-7]": 300.0}
ARM_DAMPING = {"panda_joint[1-2]": 20.0, "panda_joint[3-4]": 4.0, "panda_joint[5-7]": 2.0}
ARM_EFFORT_LIMIT = {"panda_joint[1-4]": 87.0, "panda_joint[5-7]": 12.0}  # unchanged FRANKA_PANDA_CFG values

# start (home) pose of the pouring task
ARM_HOME = (-1.07505691, 0.76868522, 0.53213346, -2.93226814, 2.50670838, 1.40047050, 0.21146376)
GRIPPER_OPEN = 0.04  # m per finger


def _per_joint(patterns: dict) -> dict[int, float]:
    """{'panda_joint[1-2]': v, ...} -> {1: v, 2: v, ...} (only the simple [a-b] patterns used above)."""
    out = {}
    for pattern, value in patterns.items():
        lo, hi = pattern[pattern.index("[") + 1 : pattern.index("]")].split("-")
        for i in range(int(lo), int(hi) + 1):
            out[i] = value
    return out


def isaac_cfg(prim_path: str = "/World/Robot"):
    """Isaac Lab ArticulationCfg of this Panda (needs Isaac Lab).

    Note: the training additionally swaps in its own spawn function to add arm collision proxies and tweaks
    the PhysX articulation settings; those training-specific details are not part of this v1.
    """
    from isaaclab.utils import clone
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

    robot = clone(FRANKA_PANDA_CFG)
    robot.prim_path = prim_path
    robot.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/{USD_ASSET_PATH}"
    robot.spawn.variants = dict(USD_VARIANTS)
    robot.spawn.rigid_props.disable_gravity = not GRAVITY_ON_ROBOT
    robot.actuators["panda_arm"].stiffness = dict(ARM_STIFFNESS)
    robot.actuators["panda_arm"].damping = dict(ARM_DAMPING)
    robot.init_state.joint_pos.update({f"panda_joint{i}": q for i, q in enumerate(ARM_HOME, start=1)})
    robot.init_state.joint_pos["panda_finger_joint.*"] = GRIPPER_OPEN
    return robot


def mujoco_settings(prefix: str = "") -> dict:
    """The same settings for a MuJoCo Menagerie franka_emika_panda/panda.xml (joint1..7, actuator1..7):
    start pose, arm servo (stiffness, damping, force limit) and gravity off. Physics engines differ, so
    this matches the robot and its servo values, not the exact motion."""
    kp, kd, force = _per_joint(ARM_STIFFNESS), _per_joint(ARM_DAMPING), _per_joint(ARM_EFFORT_LIMIT)
    targets = {f"{prefix}joint{i}": q for i, q in enumerate(ARM_HOME, start=1)}
    targets.update({f"{prefix}finger_joint1": GRIPPER_OPEN, f"{prefix}finger_joint2": GRIPPER_OPEN})
    return dict(
        joint_targets=targets,
        gains={f"{prefix}actuator{i}": (kp[i], kd[i], force[i]) for i in range(1, 8)},
        gravcomp=not GRAVITY_ON_ROBOT,
    )
