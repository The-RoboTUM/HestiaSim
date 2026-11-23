# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the kitchen beans-pour task (``HestiaBeansPour-Franka-Kitchen-Pour``).

Reused from the pour task (``HestiaBeansPour-Franka-Pour``) unchanged: the Franka Panda (same USD, colliders, gains,
gravity off), the source container with its beans, the action space, the PhysX settings and many MDP terms.
New: the Hestia kitchen (``scenes/u_table_layout``), the mixer with its chute as the target, starting from the robot's
home pose without a reset dataset, a camera-free (proprioceptive) actor, and noise for sim-to-real.
"""

from __future__ import annotations

import math

from isaaclab_visualizers.kit import KitBackendCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils.presets import MultiBackendVisualizerCfg

from ..pour.beans import BeanShape, bean_bounding_radius, bean_lattice_c, build_beans_collection_cfg
from ..pour.geometry import SOURCE_CUP_GEOMETRY
from ..pour.pour_env_cfg import _robot_cfg, _source_cup_asset_cfg
from . import mdp
from .layout import MIXER_COLLIDER_BOXES, kitchen_layout, write_kitchen_statics_usda

LAYOUT = kitchen_layout()
KITCHEN_STATICS_USD = write_kitchen_statics_usda()

# Beans: the pour task's beans (~1 cm spheres, 0.6 g). "Half full": 5 layers of 11 mm = a 55 mm bean column in the
# 110 mm deep container (45 beans). bean_lattice_c counts layers that fit in fill_level * depth incl. 6 mm clearance,
# so 0.55 is the level that gives exactly the 5 layers (0.50 would give 4 layers = 40 %).
_BEAN_SHAPE: BeanShape = "sphere"
_BEAN_RADIUS = 0.005
_BEAN_LENGTH = 0.004
_BEAN_GAP = 0.001
_BEAN_MASS = 0.6e-3
_BEAN_FRICTION = 0.5
_BEAN_RESTITUTION = 0.05
_BEAN_COLOR = (0.55, 0.33, 0.16)
_BEAN_FILL_LEVEL = 0.55


##
# Spawners
##


def spawn_mixer(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the mixer USD (Mana v1) as one kinematic body with simplified box colliders.

    The USD's own physics (rigid bowl and lid, convex-decomposition colliders) is removed: convex decomposition
    would close the chute hole, and nested rigid bodies are not allowed in a rigid object. The boxes of
    :data:`layout.MIXER_COLLIDER_BOXES` (machine, bowl, lid with the hole, chute walls) are added instead; the
    meshes stay for looks.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

    mixer_prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for root_prim in sim_utils.find_matching_prims(prim_path, stage=mixer_prim.GetStage()):
        stage = root_prim.GetStage()
        for prim in list(Usd.PrimRange(root_prim)):
            for api in (
                UsdPhysics.RigidBodyAPI,
                UsdPhysics.MassAPI,
                UsdPhysics.MeshCollisionAPI,
                UsdPhysics.CollisionAPI,
            ):
                if prim.HasAPI(api):
                    prim.RemoveAPI(api)
        UsdPhysics.RigidBodyAPI.Apply(root_prim).CreateKinematicEnabledAttr(True)
        colliders_path = root_prim.GetPath().AppendChild("box_colliders")
        UsdGeom.Xform.Define(stage, colliders_path)
        for name, centre, half_extents, pitch_deg in MIXER_COLLIDER_BOXES:
            cube = UsdGeom.Cube.Define(stage, colliders_path.AppendChild(name))
            cube.CreateSizeAttr(1.0)
            cube.ClearXformOpOrder()
            cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
            cube.AddRotateYOp().Set(float(pitch_deg))
            cube.AddScaleOp().Set(Gf.Vec3f(*(2.0 * value for value in half_extents)))
            cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
    return mixer_prim


##
# Scene assets
##


def _kitchen_robot_cfg():
    """The pour task's Franka, mounted like in the scene (on the adapter, facing south-east), in its home pose."""
    robot = _robot_cfg()
    robot.init_state.pos = LAYOUT.robot_pos
    robot.init_state.rot = LAYOUT.robot_rot
    robot.init_state.joint_pos.update({f"panda_joint{i}": q for i, q in enumerate(LAYOUT.robot_home, start=1)})
    robot.init_state.joint_pos["panda_finger_joint.*"] = LAYOUT.gripper_open
    return robot


def _kitchen_cup_cfg() -> RigidObjectCfg:
    """The pour task's source container, standing at its spot on the middle table."""
    cup = _source_cup_asset_cfg()
    cup.init_state.pos = LAYOUT.cup_pos
    cup.init_state.rot = LAYOUT.cup_rot
    return cup


def _mixer_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Mixer",
        init_state=RigidObjectCfg.InitialStateCfg(pos=LAYOUT.mixer_pos, rot=LAYOUT.mixer_rot),
        spawn=UsdFileCfg(usd_path=LAYOUT.mixer_usd, func=spawn_mixer),
    )


def _default_beans_cfg() -> RigidObjectCollectionCfg:
    positions = bean_lattice_c(
        SOURCE_CUP_GEOMETRY,
        bounding_radius=bean_bounding_radius(_BEAN_SHAPE, _BEAN_RADIUS, _BEAN_LENGTH),
        gap=_BEAN_GAP,
        fill_level=_BEAN_FILL_LEVEL,
    )
    return build_beans_collection_cfg(
        positions,
        cup_pos=LAYOUT.cup_pos,
        shape=_BEAN_SHAPE,
        radius=_BEAN_RADIUS,
        length=_BEAN_LENGTH,
        mass=_BEAN_MASS,
        friction=_BEAN_FRICTION,
        restitution=_BEAN_RESTITUTION,
        color=_BEAN_COLOR,
    )


@configclass
class KitchenPourSceneCfg(InteractiveSceneCfg):
    """The Hestia kitchen: U tables, the Franka on its adapter, the mixer, the bean container."""

    # Static box colliders of the three tables + the Panda adapter, generated from scenes/u_table_layout/u_layout.py
    # (layout.write_kitchen_statics_usda). The real table meshes are not needed without cameras and would be much
    # more expensive to collide with in 1000+ environments.
    kitchen = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Kitchen", spawn=UsdFileCfg(usd_path=KITCHEN_STATICS_USD))
    plane = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    robot = _kitchen_robot_cfg()
    mixer: RigidObjectCfg = _mixer_cfg()
    source_cup: RigidObjectCfg = _kitchen_cup_cfg()
    beans: RigidObjectCollectionCfg = _default_beans_cfg()


##
# MDP
##


@configclass
class ActionsCfg:
    """The pour task's actions: filtered relative arm joint deltas (+ actuation noise) and a binary gripper."""

    arm_action = mdp.NoisyEMARelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[f"panda_joint{index}" for index in range(1, 8)],
        preserve_order=True,
        scale=0.03,
        use_zero_offset=True,
        alpha=0.20,
        noise_std=0.002,
    )
    gripper_action = mdp.CurriculumGripperPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        alpha=1.0 - (1.0 - 0.2) ** (1.0 / 3.0),
        close_position=0.021,
        neutral_position=0.04,
        default_position=0.04,  # episodes start with the hand open
        contact_min_deflection=0.001,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Actor: only what the real robot measures (no cameras), with sensor noise."""

        arm_q = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
            noise=Unoise(n_min=-0.003, n_max=0.003),
            scale=0.3,
        )
        arm_qd = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
            noise=Unoise(n_min=-0.03, n_max=0.03),
            scale=0.05,
        )
        finger_position = ObsTerm(func=mdp.finger_position_obs, noise=Unoise(n_min=-0.0005, n_max=0.0005), scale=25.0)
        gripper_target = ObsTerm(func=mdp.gripper_target_obs, scale=25.0)
        last_action = ObsTerm(func=mdp.last_action, scale=0.2)
        time_remaining = ObsTerm(func=mdp.time_remaining_obs)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged state for the critic only (not needed on the robot)."""

        tcp_pose = ObsTerm(func=mdp.tcp_pose_b_obs)
        cup_pose = ObsTerm(func=mdp.cup_pose_b_obs)
        mixer_pose = ObsTerm(func=mdp.mixer_pose_b_obs)
        tcp_to_grasp = ObsTerm(func=mdp.tcp_to_grasp_c_obs, scale=5.0)
        grasp_to_tcp_quat = ObsTerm(func=mdp.grasp_to_tcp_quat_obs)
        mouth_to_target = ObsTerm(func=mdp.mouth_to_target_b_obs, scale=2.0)
        cup_tilt = ObsTerm(func=mdp.cup_tilt_obs)
        cup_velocity = ObsTerm(
            func=mdp.normalized_cup_velocity_obs,
            params={"max_surface_speed": 0.5, "surface_radius": 0.13},
            clip=(-2.0, 2.0),
        )
        gripper_contact = ObsTerm(func=mdp.gripper_contact_obs, scale=250.0)
        bean_fractions = ObsTerm(func=mdp.bean_fractions_obs)
        task_stage = ObsTerm(func=mdp.task_stage_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Staged potential (shaping) + bean transfer, terminal success / failure, smoothness regularizers."""

    progress = RewTerm(func=mdp.DeltaReward, weight=2.0, params={"quantity": "potential"})
    delivered = RewTerm(func=mdp.DeltaReward, weight=4.0, params={"quantity": "delivered"})
    spilled = RewTerm(func=mdp.DeltaReward, weight=-4.0, params={"quantity": "spilled"})
    success = RewTerm(func=mdp.pour_success_bonus, weight=5.0)
    failure = RewTerm(func=mdp.terminal_failure, weight=-1.0, params={"include_time_out": False})
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-1.0e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5.0e-4)
    joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )


@configclass
class TerminationsCfg:
    """Failures first (they take precedence over success), then success, then the neutral timeout."""

    failure = DoneTerm(func=mdp.nonfinite_failure)
    extreme_rigid_state = DoneTerm(func=mdp.extreme_rigid_state)
    cup_dropped = DoneTerm(func=mdp.cup_dropped)
    spill = DoneTerm(func=mdp.excessive_spill, params={"terminate": True})
    bean_out_of_bounds = DoneTerm(func=mdp.bean_out_of_bounds)
    success = DoneTerm(func=mdp.immediate_pour_success, params={})
    time_out = DoneTerm(func=mdp.unsuccessful_time_out, time_out=True)


@configclass
class EventsCfg:
    reset_scene = EventTerm(func=mdp.reset_kitchen_scene, mode="reset")


##
# Environment
##


@configclass
class FrankaKitchenPourEnvCfg(ManagerBasedRLEnvCfg):
    """Franka pours a half-full container of beans from the middle table into the mixer's chute (PhysX)."""

    class_type: type | str = "hestia_beans_pour.tasks.kitchen_pour.kitchen_env:FrankaKitchenPourEnv"

    scene: KitchenPourSceneCfg = KitchenPourSceneCfg(num_envs=1024, env_spacing=5.0)
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        visualizer_cfgs=VisualizerCfg(
            eye=(0.2, 1.9, 1.7),
            lookat=(1.0, 0.55, 0.8),
            backend=MultiBackendVisualizerCfg(kit=KitBackendCfg(origin_type="env", origin_env_index=0)),
        )
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    tcp_body_name: str = "panda_hand"
    tcp_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.107)
    tcp_offset_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    table_top: float = LAYOUT.table_top
    """Height of the table tops in the env frame [m]."""

    # -- beans (same meaning as in the pour task; can be overridden from the command line)
    bean_shape: BeanShape = _BEAN_SHAPE
    bean_radius: float = _BEAN_RADIUS
    bean_length: float = _BEAN_LENGTH
    bean_gap: float = _BEAN_GAP
    bean_mass: float = _BEAN_MASS
    bean_friction: float = _BEAN_FRICTION
    bean_restitution: float = _BEAN_RESTITUTION
    bean_color: tuple[float, float, float] = _BEAN_COLOR
    bean_fill_level: float = _BEAN_FILL_LEVEL
    """Half full, see the comment at ``_BEAN_FILL_LEVEL``."""
    num_beans: int | None = None
    bean_reset_jitter: float = 0.0004

    pour_target_frac: float = 0.85
    """Fraction of the beans that must be in the mixer (chute or bowl) for success."""
    max_spill_fraction: float = 0.15
    """More spilled beans than this ends the episode as a failure."""
    bean_count_margin: float = 0.003
    spill_height_above_table: float = 0.02
    """A bean lower than this above the table tops (on a table, the adapter or the floor) is spilled [m]."""
    spill_rest_speed: float = 0.05
    """A bean resting (slower than this [m/s]) on the mixer outside the chute is spilled."""

    # -- noise for sim-to-real (uniform, +-)
    reset_joint_noise: float = 0.03
    """Arm start pose around home [rad]."""
    cup_pos_noise: float = 0.010
    """Container placement on the middle table [m]: +-1 cm in x and y."""
    cup_yaw_noise: float = math.radians(10.0)
    """Container rotation [rad]."""
    mixer_pos_noise: float = 0.005
    """Mixer placement [m] (sim model vs. real mixer)."""
    mixer_yaw_noise: float = math.radians(2.0)

    # -- safety bounds (env frame), used by the reused pour-task terminations
    state_bound_joint_position_margin: float = 0.05
    state_bound_max_joint_velocity: float = 20.0
    state_bound_max_cup_linear_velocity: float = 10.0
    state_bound_max_cup_angular_velocity: float = 50.0
    workspace_lower_bound: tuple[float, float, float] = (-1.0, -1.0, -0.2)
    workspace_upper_bound: tuple[float, float, float] = (2.5, 2.0, 2.2)

    def __post_init__(self):
        """30 Hz control, 15 s episodes, 240 Hz PhysX (same physics settings as the pour task)."""
        self.decimation = 8
        self.episode_length_s = 15.0
        self.sim.dt = 1.0 / 240.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            solver_type=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**24,
            gpu_max_rigid_patch_count=2**22,
            gpu_found_lost_pairs_capacity=2**23,
            gpu_found_lost_aggregate_pairs_capacity=2**26,
            gpu_total_aggregate_pairs_capacity=2**23,
        )

    def validate_config(self) -> None:
        """Catch bad command-line overrides early."""
        for name in ("bean_fill_level", "pour_target_frac", "max_spill_fraction"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1].")
        if self.bean_shape not in ("sphere", "capsule"):
            raise ValueError("bean_shape must be 'sphere' or 'capsule'.")
        for name in (
            "bean_reset_jitter",
            "reset_joint_noise",
            "cup_pos_noise",
            "cup_yaw_noise",
            "mixer_pos_noise",
            "mixer_yaw_noise",
        ):
            if not math.isfinite(float(getattr(self, name))) or float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.bean_reset_jitter > 0.0 and 2.0 * self.bean_reset_jitter >= self.bean_gap:
            raise ValueError("bean_reset_jitter must be smaller than bean_gap / 2.")
        if self.cup_pos_noise > 0.02:
            raise ValueError("cup_pos_noise above 2 cm exceeds what a blind grasp can tolerate (gripper opening 8 cm).")

    def play_mode(self) -> None:
        """One environment, camera on the robot and the mixer."""
        super().play_mode()
        self.scene.num_envs = 1
        self.sim.visualizer_cfgs.eye = (0.55, 1.35, 1.35)
