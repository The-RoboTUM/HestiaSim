# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for reset-dataset Franka pouring of rigid beans with Isaac Sim PhysX.

This task mirrors Isaac Lab's ``IsaacContrib-Franka-Pour`` (Newton MJWarp + MPM media): the same Franka
Panda, SeattleLab table, source and receiver cups, action space, reset curriculum, and terminal-sparse
reward. The granular MPM media is replaced by a configurable number of small rigid beans, and the whole
scene is simulated by PhysX inside Isaac Sim.
"""

from __future__ import annotations

import math
import os
import struct
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

from isaaclab_visualizers.kit import KitBackendCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas import MassCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import clone, configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.visualizers import VisualizerCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg

from isaaclab_tasks.utils.presets import MultiBackendVisualizerCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from . import mdp
from .beans import BeanShape, bean_bounding_radius, bean_lattice_c, build_beans_collection_cfg
from .geometry import (
    CUP_GRASP_HEIGHT,
    CUP_GRASP_TCP_QUAT_C,
    SOURCE_CUP_GEOMETRY,
    TARGET_CUP_GEOMETRY,
    CupGeometry,
)
from .reset_sampler import ResetDatasetSamplerCfg

# The robot, cups, and reset dataset are the published Isaac Lab Franka Pour assets.
FRANKA_POUR_ROBOT_ASSET_PATH = "Robots/FrankaEmika/franka_panda.usda"
FRANKA_POUR_CUPS_ASSET_PATH = "Contrib/MPM/Pour/franka_pour_cups.usda"
FRANKA_POUR_RESET_DATASET_ASSET_PATH = "Contrib/MPM/Pour/reset_dataset.pt"
FRANKA_POUR_ROBOT_ASSET_ID = f"{ISAACLAB_NUCLEUS_DIR}/{FRANKA_POUR_ROBOT_ASSET_PATH}"
FRANKA_POUR_CUPS_ASSET_ID = f"{ISAACLAB_NUCLEUS_DIR}/{FRANKA_POUR_CUPS_ASSET_PATH}"
FRANKA_POUR_RESET_DATASET_ASSET_ID = f"{ISAACLAB_NUCLEUS_DIR}/{FRANKA_POUR_RESET_DATASET_ASSET_PATH}"
FRANKA_POUR_ROBOT_USD_PATH = os.environ.get("ISAACLAB_FRANKA_POUR_ROBOT_USD_PATH", FRANKA_POUR_ROBOT_ASSET_ID)
FRANKA_POUR_CUPS_USD_PATH = os.environ.get("ISAACLAB_FRANKA_POUR_CUPS_USD_PATH", FRANKA_POUR_CUPS_ASSET_ID)

FRANKA_POUR_ARM_DRIVE_STIFFNESS = MappingProxyType(
    {
        "panda_joint[1-2]": 1000.0,
        "panda_joint[3-4]": 750.0,
        "panda_joint[5-7]": 300.0,
    }
)
FRANKA_POUR_ARM_DRIVE_DAMPING = MappingProxyType(
    {
        "panda_joint[1-2]": 20.0,
        "panda_joint[3-4]": 4.0,
        "panda_joint[5-7]": 2.0,
    }
)
FRANKA_POUR_ARM_COLLISION_PROXIES = frozenset(
    {
        "link0_c",
        "link1_c",
        "link2_c",
        "link3_c",
        "link4_c",
        "link5_c0",
        "link5_c1",
        "link5_c2",
        "link6_c",
        "link7_c",
    }
)

_ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
_ARM_HOME = (
    -1.07505691,
    0.76868522,
    0.53213346,
    -2.93226814,
    2.50670838,
    1.40047050,
    0.21146376,
)
_GRIPPER_OPEN_POSITION = 0.04
_SOURCE_CUP_POSITION = (0.5, 0.0, 0.0)
_TARGET_CUP_POSITION = (0.5, -0.18, 0.0)
_SOURCE_CUP_MASS = 0.05
_SOURCE_CUP_GRASP_FRICTION = 2.0
_TARGET_CUP_FRICTION = 0.8

# Bean defaults: ~1 cm round beans (chickpea / coffee-bean scale) filling 70% of the source cup.
_BEAN_SHAPE: BeanShape = "sphere"
_BEAN_RADIUS = 0.005
_BEAN_LENGTH = 0.004
_BEAN_GAP = 0.001
_BEAN_MASS = 0.6e-3
_BEAN_FRICTION = 0.5
_BEAN_RESTITUTION = 0.05
_BEAN_COLOR = (0.55, 0.33, 0.16)
_BEAN_FILL_LEVEL = 0.70


##
# Spawners
##


def spawn_franka_with_arm_collisions(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the canonical Franka with its convex-hull arm colliders enabled."""
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    robot_prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for root_prim in sim_utils.find_matching_prims(prim_path, stage=robot_prim.GetStage()):
        proxy_roots = {
            prim.GetName(): prim.GetParent().GetPath()
            for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies())
            if prim.GetName() in FRANKA_POUR_ARM_COLLISION_PROXIES and prim.GetParent().GetName() == prim.GetName()
        }
        if proxy_roots.keys() != FRANKA_POUR_ARM_COLLISION_PROXIES:
            missing = sorted(FRANKA_POUR_ARM_COLLISION_PROXIES.difference(proxy_roots))
            raise RuntimeError(f"Franka asset at {root_prim.GetPath()} is missing collision proxies: {missing}.")
        for proxy_root in proxy_roots.values():
            root_prim.GetStage().OverridePrim(proxy_root).SetInstanceable(False)
        for proxy_root in proxy_roots.values():
            collision_prim = root_prim.GetStage().GetPrimAtPath(proxy_root.AppendChild(proxy_root.name))
            UsdPhysics.CollisionAPI(collision_prim).GetCollisionEnabledAttr().Set(True)
    return robot_prim


def spawn_static_table(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Keep the stationary table's colliders without a rigid-body simulation state."""
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    table_prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for root_prim in sim_utils.find_matching_prims(prim_path, stage=table_prim.GetStage()):
        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
    return table_prim


_CUP_BOX_COLLIDER_NAMES = ("bottom", "wall_px", "wall_nx", "wall_py", "wall_ny")


def _replace_cup_colliders_with_boxes(root_prim, geometry: CupGeometry) -> None:
    """Swap the authored cup colliders for a hollow five-box PhysX collider.

    The published cups were authored for Newton: the source cup collides with the robot through a solid
    ``grasp_proxy`` box and with particles through a triangle-mesh ``bowl`` (not supported on a dynamic
    PhysX body); the receiver uses a solid ``rigid_collider``. Rigid beans must physically sit *inside* both
    cups, so every authored collider is removed and replaced by the cup's bottom plate and four walls. The
    outer envelope is identical to the Newton grasp proxy, so grasp states in the reset dataset still apply.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

    stage = root_prim.GetStage()
    geometry_prim = root_prim.GetChild("geometry")
    if not geometry_prim.IsValid():
        raise RuntimeError(f"Cup asset at {root_prim.GetPath()} has no 'geometry' child prim.")
    for prim in list(Usd.PrimRange(geometry_prim)):
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)

    colliders_path = geometry_prim.GetPath().AppendChild("box_colliders")
    UsdGeom.Xform.Define(stage, colliders_path)
    for name, (offset, half_extents) in zip(_CUP_BOX_COLLIDER_NAMES, geometry.collision_boxes, strict=True):
        cube = UsdGeom.Cube.Define(stage, colliders_path.AppendChild(name))
        cube.CreateSizeAttr(1.0)
        cube.ClearXformOpOrder()
        cube.AddTranslateOp().Set(Gf.Vec3d(*offset))
        cube.AddScaleOp().Set(Gf.Vec3f(*(2.0 * value for value in half_extents)))
        cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)


def _spawn_cup(prim_path, cfg, translation, orientation, geometry: CupGeometry, **kwargs):
    cup_prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for root_prim in sim_utils.find_matching_prims(prim_path, stage=cup_prim.GetStage()):
        _replace_cup_colliders_with_boxes(root_prim, geometry)
    return cup_prim


def spawn_source_cup(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the source cup with a hollow PhysX box collider."""
    return _spawn_cup(prim_path, cfg, translation, orientation, SOURCE_CUP_GEOMETRY, **kwargs)


def spawn_target_cup(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the receiver cup with a hollow PhysX box collider."""
    return _spawn_cup(prim_path, cfg, translation, orientation, TARGET_CUP_GEOMETRY, **kwargs)


##
# Scene assets
##


def _robot_cfg():
    """Franka Panda with the PhysX payload, convex-hull colliders, and the pouring drive gains."""
    robot = clone(FRANKA_PANDA_CFG)
    robot.prim_path = "{ENV_REGEX_NS}/Robot"
    robot.spawn.usd_path = FRANKA_POUR_ROBOT_USD_PATH
    robot.spawn.variants = {"Physics": "physx", "Colliders": "convex_hulls"}
    robot.spawn.func = spawn_franka_with_arm_collisions
    # The Newton task uses MuJoCo actuator gravity compensation; PhysX disables gravity on the links instead.
    robot.spawn.rigid_props.disable_gravity = True
    physx_articulation = next(frag for frag in robot.spawn.articulation_props if isinstance(frag, PhysxArticulationCfg))
    physx_articulation.enabled_self_collisions = True
    physx_articulation.solver_position_iteration_count = 16
    physx_articulation.solver_velocity_iteration_count = 1
    robot.actuators["panda_arm"].stiffness = dict(FRANKA_POUR_ARM_DRIVE_STIFFNESS)
    robot.actuators["panda_arm"].damping = dict(FRANKA_POUR_ARM_DRIVE_DAMPING)
    robot.init_state.joint_pos.update(dict(zip(_ARM_JOINT_NAMES, _ARM_HOME, strict=True)))
    robot.init_state.joint_pos["panda_finger_joint.*"] = _GRIPPER_OPEN_POSITION
    return robot


def _source_cup_asset_cfg() -> RigidObjectCfg:
    """Dynamic source cup holding the beans."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SourceCup",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_SOURCE_CUP_POSITION, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=UsdFileCfg(
            usd_path=FRANKA_POUR_CUPS_USD_PATH,
            func=spawn_source_cup,
            variants={"cupType": "source"},
            mass_props=MassCfg(mass=_SOURCE_CUP_MASS),
            rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=False),
            physics_material=RigidBodyMaterialBaseCfg(
                static_friction=_SOURCE_CUP_GRASP_FRICTION,
                dynamic_friction=_SOURCE_CUP_GRASP_FRICTION,
            ),
        ),
    )


def _target_cup_asset_cfg() -> RigidObjectCfg:
    """Kinematic receiver cup."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetCup",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_TARGET_CUP_POSITION, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=UsdFileCfg(
            usd_path=FRANKA_POUR_CUPS_USD_PATH,
            func=spawn_target_cup,
            variants={"cupType": "target"},
            rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
            physics_material=RigidBodyMaterialBaseCfg(
                static_friction=_TARGET_CUP_FRICTION,
                dynamic_friction=_TARGET_CUP_FRICTION,
            ),
        ),
    )


def beans_local_positions(cfg: FrankaBeansPourEnvCfg) -> list[tuple[float, float, float]]:
    """Resolve the bean lattice in the source-cup frame from the current configuration."""
    return bean_lattice_c(
        SOURCE_CUP_GEOMETRY,
        bounding_radius=bean_bounding_radius(cfg.bean_shape, cfg.bean_radius, cfg.bean_length),
        gap=cfg.bean_gap,
        fill_level=cfg.bean_fill_level,
        max_beans=cfg.num_beans,
    )


def configure_beans(cfg: FrankaBeansPourEnvCfg) -> None:
    """Rebuild the bean collection after command-line overrides of the bean parameters."""
    cfg.scene.beans = build_beans_collection_cfg(
        beans_local_positions(cfg),
        cup_pos=tuple(cfg.scene.source_cup.init_state.pos),
        shape=cfg.bean_shape,
        radius=cfg.bean_radius,
        length=cfg.bean_length,
        mass=cfg.bean_mass,
        friction=cfg.bean_friction,
        restitution=cfg.bean_restitution,
        color=cfg.bean_color,
    )


def _default_beans_cfg() -> RigidObjectCollectionCfg:
    """Bean collection for the default bean parameters."""
    positions = bean_lattice_c(
        SOURCE_CUP_GEOMETRY,
        bounding_radius=bean_bounding_radius(_BEAN_SHAPE, _BEAN_RADIUS, _BEAN_LENGTH),
        gap=_BEAN_GAP,
        fill_level=_BEAN_FILL_LEVEL,
    )
    return build_beans_collection_cfg(
        positions,
        cup_pos=_SOURCE_CUP_POSITION,
        shape=_BEAN_SHAPE,
        radius=_BEAN_RADIUS,
        length=_BEAN_LENGTH,
        mass=_BEAN_MASS,
        friction=_BEAN_FRICTION,
        restitution=_BEAN_RESTITUTION,
        color=_BEAN_COLOR,
    )


@configclass
class BeansPourSceneCfg(InteractiveSceneCfg):
    """Lift-table scene with the Franka, the source cup full of beans, and the receiver cup."""

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0, 0, 0.707, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            func=spawn_static_table,
            make_uninstanceable=True,
        ),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    robot = _robot_cfg()
    source_cup: RigidObjectCfg = _source_cup_asset_cfg()
    target_cup: RigidObjectCfg = _target_cup_asset_cfg()
    beans: RigidObjectCollectionCfg = _default_beans_cfg()


##
# MDP
##


@configclass
class ActionsCfg:
    """Filtered arm deltas and a binary gripper command."""

    arm_action = mdp.EMARelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[f"panda_joint{index}" for index in range(1, 8)],
        preserve_order=True,
        scale=0.03,
        use_zero_offset=True,
        alpha=0.20,
    )
    gripper_action = mdp.CurriculumGripperPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        alpha=1.0 - (1.0 - 0.2) ** (1.0 / 3.0),
        close_position=0.021,
        neutral_position=0.04,
        default_position=0.024,
        contact_min_deflection=0.001,
    )


@configclass
class ObservationsCfg:
    """Current robot and task state with compact, permutation-invariant bean summaries."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Robot, cup, and target state available to the actor."""

        arm_q = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
            scale=0.3,
        )
        arm_qd = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
            scale=0.05,
        )
        time_remaining = ObsTerm(func=mdp.time_remaining_obs)
        pour_target_fraction = ObsTerm(func=mdp.pour_target_fraction_obs)
        tcp_pose = ObsTerm(func=mdp.tcp_pose_obs)
        cup_pose = ObsTerm(func=mdp.cup_pose_obs)
        target_pose = ObsTerm(func=mdp.target_pose_obs)
        tcp_to_grasp_position_c = ObsTerm(func=mdp.tcp_to_grasp_position_c_obs, scale=10.0)
        grasp_to_tcp_quat = ObsTerm(func=mdp.grasp_to_tcp_quat_obs)
        target_position_c = ObsTerm(func=mdp.target_position_c_obs, scale=5.0)
        finger_position = ObsTerm(func=mdp.finger_position_obs, scale=25.0)
        finger_velocity = ObsTerm(func=mdp.finger_velocity_obs, scale=5.0)
        gripper_target = ObsTerm(func=mdp.gripper_target_obs, scale=25.0)
        gripper_contact = ObsTerm(func=mdp.gripper_contact_obs, scale=250.0)
        last_action = ObsTerm(func=mdp.last_action, scale=0.2)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class BeansCfg(ObsGroup):
        """Permutation-invariant bean and cup-motion state."""

        cup_velocity = ObsTerm(
            func=mdp.normalized_cup_velocity_obs,
            params={"max_surface_speed": 0.5, "surface_radius": 0.13},
            clip=(-2.0, 2.0),
        )
        bean_fractions = ObsTerm(func=mdp.bean_fractions_obs)
        bean_source_state = ObsTerm(func=mdp.bean_source_state_obs)
        bean_transfer = ObsTerm(func=mdp.bean_transfer_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 0

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Exact simulation state available only to the asymmetric critic."""

        success_dwell = ObsTerm(func=mdp.success_dwell_obs)
        lost_grasp_dwell = ObsTerm(func=mdp.lost_grasp_dwell_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    beans: BeansCfg = BeansCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class RewardsCfg:
    """Terminal-sparse pouring reward with small action regularizers."""

    success = RewTerm(func=mdp.pour_success_bonus, weight=5.0)
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-1.0e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)
    failure = RewTerm(func=mdp.terminal_failure, weight=-1.0, params={"include_time_out": False})


@configclass
class TerminationsCfg:
    """Safety failures, immediate bean-transfer success, and neutral timeout."""

    failure = DoneTerm(func=mdp.nonfinite_failure)
    extreme_rigid_state = DoneTerm(func=mdp.extreme_rigid_state)
    source_receiver_overlap = DoneTerm(func=mdp.source_receiver_overlap, params={"clearance": 0.001})
    lost_grasp = DoneTerm(
        func=mdp.lost_lifted_grasp,
        params={
            "dwell_time_s": 0.05,
            "max_tcp_distance": 0.018,
            "max_gripper_width_error": 0.006,
            "max_gripper_command": 0.027,
            "terminate": False,
        },
    )
    spill = DoneTerm(func=mdp.excessive_spill, params={"terminate": True})
    bean_out_of_bounds = DoneTerm(func=mdp.bean_out_of_bounds)
    success = DoneTerm(func=mdp.immediate_pour_success, params={})
    learning_progress_context = DoneTerm(
        func=mdp.PourResetLearningProgress,
        params={
            "minimum_progress": 0.08,
            "minimum_episode_steps": 3,
            "potential_params": {
                "approach_position_std": 0.20,
                "approach_orientation_std": 0.75,
                "approach_open_hand_fraction": 0.15,
                "grasp_target_height": 0.10,
                "grasp_reach_std": 0.025,
                "grasp_preload_position": 0.024,
                "grasp_fraction": 0.40,
                "source_mouth_height": 0.119,
                "target_rim_height": 0.074,
                "target_clearance_height": 0.15,
                "alignment_std": 0.20,
                "tilt_alignment_radius": 0.15,
                "target_tilt": math.radians(140.0),
            },
        },
    )
    time_out = DoneTerm(func=mdp.unsuccessful_time_out, time_out=True)


@configclass
class EventsCfg:
    """Reset the scene from the row selected by the curriculum term."""

    reset_scene = EventTerm(func=mdp.reset_pour_scene, mode="reset")


@configclass
class CurriculumCfg:
    """Adaptive replay over the validated external reset dataset."""

    reset_dataset = CurrTerm(func=mdp.PourResetDatasetCurriculum)


##
# Environment
##


@configclass
class FrankaBeansPourEnvCfg(ManagerBasedRLEnvCfg):
    """Franka pouring rigid beans between two cups, simulated with Isaac Sim PhysX."""

    class_type: type | str = "{DIR}.pour_env:FrankaBeansPourEnv"

    scene: BeansPourSceneCfg = BeansPourSceneCfg(num_envs=1024, env_spacing=2.5)
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        visualizer_cfgs=VisualizerCfg(
            eye=(1.4, 1.4, 0.9),
            lookat=(0.5, 0.0, 0.1),
            backend=MultiBackendVisualizerCfg(kit=KitBackendCfg(origin_type="env", origin_env_index=0)),
        )
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    tcp_body_name: str = "panda_hand"
    tcp_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.107)
    tcp_offset_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    # -- beans
    bean_shape: BeanShape = _BEAN_SHAPE
    """Bean primitive: ``"sphere"`` or ``"capsule"`` (elongated, closer to real beans)."""
    bean_radius: float = _BEAN_RADIUS
    """Sphere or capsule radius [m]."""
    bean_length: float = _BEAN_LENGTH
    """Capsule cylinder length [m]; ignored for spheres."""
    bean_gap: float = _BEAN_GAP
    """Clearance between beans and between beans and the cup walls at reset [m]."""
    bean_mass: float = _BEAN_MASS
    """Mass of one bean [kg]."""
    bean_friction: float = _BEAN_FRICTION
    bean_restitution: float = _BEAN_RESTITUTION
    bean_color: tuple[float, float, float] = _BEAN_COLOR
    bean_fill_level: float = _BEAN_FILL_LEVEL
    """Initial fill height as a fraction of the source-cup cavity height, in ``(0, 1]``."""
    num_beans: int | None = None
    """Exact number of beans. ``None`` fills the lattice up to :attr:`bean_fill_level`."""
    bean_reset_jitter: float = 0.0004
    """Uniform horizontal jitter applied to each lattice position at reset [m]. Must stay below ``bean_gap / 2``."""

    pour_target_frac: float = 0.70
    """Fraction of the beans that must be inside the receiver for success."""

    bean_count_margin: float = 0.003
    spill_table_height: float = 0.0
    max_spill_fraction: float = 0.10
    success_dwell_time_s: float = 0.15
    success_min_lift_height: float = 0.05

    state_bound_joint_position_margin: float = 0.05
    state_bound_max_joint_velocity: float = 20.0
    state_bound_max_cup_linear_velocity: float = 10.0
    state_bound_max_cup_angular_velocity: float = 50.0
    workspace_lower_bound: tuple[float, float, float] = (-1.0, -1.0, -1.10)
    workspace_upper_bound: tuple[float, float, float] = (1.5, 1.0, 1.5)

    reset_dataset_path: str = FRANKA_POUR_RESET_DATASET_ASSET_ID
    reset_dataset_content_sha256: str | None = None
    reset_dataset_top_grasp_count: int | None = None
    reset_dataset_sampling_mode: Literal["adaptive", "uniform"] = "adaptive"
    reset_dataset_sampler: ResetDatasetSamplerCfg = ResetDatasetSamplerCfg(
        monitored_history_len=50,
        target_success_rate=0.50,
        kappa=1.0,
        epsilon=1.0e-4,
        # Reserve 50% of assignments for cyclic replay.
        uniform_fraction=0.50,
    )
    curriculum_freeze: bool = False

    def __post_init__(self):
        """Set the 30 Hz control rate, horizon, and PhysX solver."""
        # 240 Hz physics keeps ~1 cm beans from tunnelling through the 7 mm cup walls while pouring.
        self.decimation = 8
        self.episode_length_s = 12.0
        self.is_finite_horizon = False
        self.sim.dt = 1.0 / 240.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            solver_type=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            # Every environment holds dozens of mutually touching beans; size the GPU contact buffers for them.
            gpu_max_rigid_contact_count=2**24,
            gpu_max_rigid_patch_count=2**22,
            gpu_found_lost_pairs_capacity=2**23,
            gpu_found_lost_aggregate_pairs_capacity=2**26,
            gpu_total_aggregate_pairs_capacity=2**23,
        )

    def validate_config(self) -> None:
        """Validate runtime values that otherwise fail with opaque errors."""
        self._validate_runtime_values()
        self._validate_mdp_term_values()
        self._validate_reset_dataset_values()

    def _validate_runtime_values(self) -> None:
        """Validate bean, success, and workspace values consumed directly at runtime."""
        for name, value in (
            ("bean_fill_level", self.bean_fill_level),
            ("pour_target_frac", self.pour_target_frac),
        ):
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 < float(value) <= 1.0
            ):
                raise ValueError(f"{name} must lie in (0, 1].")
        if self.bean_shape not in ("sphere", "capsule"):
            raise ValueError("bean_shape must be 'sphere' or 'capsule'.")
        for name in ("bean_radius", "bean_mass"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("bean_length", "bean_gap", "bean_friction", "bean_restitution", "bean_reset_jitter"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if 2.0 * float(self.bean_reset_jitter) >= float(self.bean_gap) and self.bean_reset_jitter > 0.0:
            raise ValueError("bean_reset_jitter must be smaller than bean_gap / 2 to keep resets collision-free.")
        if self.num_beans is not None and (
            isinstance(self.num_beans, bool) or not isinstance(self.num_beans, int) or self.num_beans <= 0
        ):
            raise ValueError("num_beans must be a positive integer or None.")
        # Raises if the requested beans do not fit.
        beans_local_positions(self)
        lower = self.workspace_lower_bound
        upper = self.workspace_upper_bound
        if len(lower) != 3 or len(upper) != 3 or any(not math.isfinite(value) for value in (*lower, *upper)):
            raise ValueError("workspace bounds must each contain three finite values.")
        if any(low >= high for low, high in zip(lower, upper, strict=True)):
            raise ValueError("workspace lower bounds must be smaller than upper bounds.")

    def _validate_mdp_term_values(self) -> None:
        """Validate constant manager-term parameters once instead of on every policy step."""

        def params(term: Any, name: str) -> Mapping[str, Any]:
            value = term.params
            if not isinstance(value, Mapping):
                raise TypeError(f"{name}.params must be a mapping.")
            return value

        def numbers(
            values: Mapping[str, Any],
            names: tuple[str, ...],
            lower: float = -math.inf,
            upper: float = math.inf,
            *,
            lower_open: bool = False,
            upper_open: bool = False,
        ) -> None:
            for name in names:
                value = values.get(name)
                if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
                    raise ValueError(f"{name} must be finite.")
                value = float(value)
                lower_valid = value > lower if lower_open else value >= lower
                upper_valid = value < upper if upper_open else value <= upper
                if not lower_valid or not upper_valid:
                    left, right = ("(" if lower_open else "["), (")" if upper_open else "]")
                    raise ValueError(f"{name} must lie in {left}{lower}, {upper}{right}.")

        velocity = params(self.observations.beans.cup_velocity, "observations.beans.cup_velocity")
        numbers(velocity, ("max_surface_speed", "surface_radius"), 0.0, lower_open=True)

        failure = params(self.rewards.failure, "rewards.failure")
        overlap = params(self.terminations.source_receiver_overlap, "terminations.source_receiver_overlap")
        lost_grasp = params(self.terminations.lost_grasp, "terminations.lost_grasp")
        spill = params(self.terminations.spill, "terminations.spill")
        for values, name, label in (
            (failure, "include_time_out", "include_time_out"),
            (lost_grasp, "terminate", "lost_grasp.terminate"),
            (spill, "terminate", "spill.terminate"),
        ):
            if not isinstance(values.get(name), bool):
                raise TypeError(f"{label} must be a bool.")
        numbers(overlap, ("clearance",), 0.0)
        numbers(lost_grasp, ("dwell_time_s",), 0.0, lower_open=True)
        numbers(
            lost_grasp,
            ("max_tcp_distance", "max_gripper_width_error", "max_gripper_command"),
            0.0,
        )

        progress = params(self.terminations.learning_progress_context, "terminations.learning_progress_context")
        numbers(progress, ("minimum_progress",), 0.0, 1.0, lower_open=True, upper_open=True)
        minimum_steps = progress.get("minimum_episode_steps")
        if not isinstance(minimum_steps, int) or isinstance(minimum_steps, bool):
            raise TypeError("minimum_episode_steps must be an integer.")
        if minimum_steps < 0:
            raise ValueError("minimum_episode_steps must be nonnegative.")
        potential = progress.get("potential_params")
        if not isinstance(potential, Mapping):
            raise TypeError("potential_params must be a mapping.")
        numbers(
            potential,
            (
                "approach_position_std",
                "approach_orientation_std",
                "grasp_target_height",
                "grasp_reach_std",
                "target_clearance_height",
                "alignment_std",
                "tilt_alignment_radius",
            ),
            0.0,
            lower_open=True,
        )
        numbers(potential, ("source_mouth_height", "target_rim_height"), 0.0)
        numbers(potential, ("grasp_preload_position",))
        numbers(potential, ("approach_open_hand_fraction", "grasp_fraction"), 0.0, 1.0)
        numbers(potential, ("target_tilt",), 0.0, math.pi, lower_open=True, upper_open=True)

    def _validate_reset_dataset_values(self) -> None:
        """Validate the external reset-dataset and sampler contract."""
        if not isinstance(self.reset_dataset_path, str) or not self.reset_dataset_path:
            raise ValueError("reset_dataset_path must be a non-empty string.")
        if self.reset_dataset_sampling_mode not in ("adaptive", "uniform"):
            raise ValueError("reset_dataset_sampling_mode must be 'adaptive' or 'uniform'.")
        if self.reset_dataset_content_sha256 is not None and not _is_sha256(self.reset_dataset_content_sha256):
            raise ValueError("reset_dataset_content_sha256 must be a lowercase SHA-256 digest or None.")
        if not isinstance(self.curriculum_freeze, bool):
            raise TypeError("curriculum_freeze must be a bool.")
        if self.reset_dataset_top_grasp_count is not None:
            if (
                not isinstance(self.reset_dataset_top_grasp_count, int)
                or isinstance(self.reset_dataset_top_grasp_count, bool)
                or self.reset_dataset_top_grasp_count <= 0
            ):
                raise ValueError("reset_dataset_top_grasp_count must be a positive integer or None.")
            if not self.curriculum_freeze:
                raise ValueError("reset_dataset_top_grasp_count requires curriculum_freeze=True.")
        self.reset_dataset_sampler.validate_values()

    def play_mode(self) -> None:
        """Configure frozen single-world dataset playback."""
        super().play_mode()
        self.scene.num_envs = 1
        self.sim.visualizer_cfgs.eye = (0.9, 0.65, 0.5)
        self.curriculum_freeze = True


def _float32(value: float) -> float:
    """Round one Python float to the representation stored by a float32 tensor."""
    return struct.unpack("=f", struct.pack("=f", float(value)))[0]


def reset_dataset_task_contract(cfg: FrankaBeansPourEnvCfg) -> dict[str, Any]:
    """Return the rigid-state subset of the Franka Pour reset-dataset contract.

    The published dataset was generated for the Newton task. Only the robot, cup, TCP, and gripper fields
    are needed to replay its rigid start states; Newton/MPM-specific fields are intentionally not checked.
    """
    source_half = tuple(_float32(value) for value in SOURCE_CUP_GEOMETRY.outer_half_extents)
    target_half = tuple(_float32(value) for value in TARGET_CUP_GEOMETRY.outer_half_extents)
    gripper_action = cfg.actions.gripper_action
    gripper_open_pos = cfg.scene.robot.init_state.joint_pos["panda_finger_joint.*"]
    return {
        "robot_asset": FRANKA_POUR_ROBOT_ASSET_PATH,
        "source_box_half": source_half,
        "source_mouth_height": SOURCE_CUP_GEOMETRY.rim_height,
        "target_box_half": target_half,
        "target_rim_height": float(2.0 * target_half[2]),
        "cup_grasp_tcp_quat_c": CUP_GRASP_TCP_QUAT_C,
        "cup_grasp_height": CUP_GRASP_HEIGHT,
        "tcp_body_name": str(cfg.tcp_body_name),
        "tcp_offset_pos": tuple(float(value) for value in cfg.tcp_offset_pos),
        "tcp_offset_rot": tuple(float(value) for value in cfg.tcp_offset_rot),
        "gripper_open_pos": float(gripper_open_pos),
        "gripper_grasp_reset_target": float(gripper_action.close_position),
        "gripper_contact_min_deflection": float(gripper_action.contact_min_deflection),
    }


def _is_sha256(value: object) -> bool:
    """Return whether ``value`` is a lowercase hexadecimal SHA-256 digest."""
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
