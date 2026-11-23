# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL env: the Franka of the pour task pours beans into the mixer of the Hestia kitchen (PhysX).

Scene (env frame = U-layout world frame, see :mod:`.layout`): three aluprofil tables in a U, the Franka Panda on its
adapter plate on the left table in the scene's mounting pose, the mixer (kinematic, chute hole open) in the middle of
the left table, and the pour task's source container half full of rigid beans at a fixed spot on the middle table.

Every reset puts the robot in its home pose and the container and mixer at their spots, each with a little noise
(sim-to-real robustness; the policy never observes them). Helper names mirror the pour env (``cup_pose_e``,
``tcp_pose_e``, ``_bean_region_masks``, ...) so its observation, reward and termination terms are reused unchanged.
Unlike in the pour env, ``*_e`` poses are all in the env frame; ``pose_e_to_b`` converts to the robot base frame.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import index_fill_
from isaaclab.utils import math as math_utils

from ..pour.beans import bean_bounding_radius
from ..pour.geometry import CUP_GRASP_HEIGHT, CUP_GRASP_TCP_QUAT_C, SOURCE_CUP_GEOMETRY, points_inside_box
from ..pour.pour_env_cfg import beans_local_positions, configure_beans
from .layout import (
    CHUTE_TOP_CENTRE,
    MIXER_BBOX_REGION,
    MIXER_BOWL_REGION,
    MIXER_CHUTE_REGION,
    POUR_TARGET_ABOVE_CHUTE,
)

if TYPE_CHECKING:
    from isaaclab.assets import RigidObjectCollection

    from .kitchen_env_cfg import FrankaKitchenPourEnvCfg

logger = logging.getLogger(__name__)

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]


class FrankaKitchenPourEnv(ManagerBasedRLEnv):
    """Franka grasping the bean container on the middle table and pouring the beans into the mixer's chute."""

    cfg: FrankaKitchenPourEnvCfg

    def __init__(self, cfg: FrankaKitchenPourEnvCfg, render_mode: str | None = None, **kwargs):
        # bean count / size can be overridden from the command line: rebuild the bean collection first
        configure_beans(cfg)
        super().__init__(cfg, render_mode, **kwargs)

    def load_managers(self) -> None:
        self._setup_task()
        super().load_managers()

    """
    Setup.
    """

    def _setup_task(self) -> None:
        dev = self.device
        cfg = self.cfg
        self._robot = self.scene["robot"]
        self._source_cup = self.scene["source_cup"]
        self._mixer = self.scene["mixer"]
        self._beans: RigidObjectCollection = self.scene["beans"]
        self.env_origins = self.scene.env_origins.to(device=dev, dtype=torch.float32)

        arm_joint_ids, _ = self._robot.find_joints(ARM_JOINTS, preserve_order=True, as_proxy=True)
        finger_joint_ids, _ = self._robot.find_joints(FINGER_JOINTS, preserve_order=True, as_proxy=True)
        self._arm_joint_ids = arm_joint_ids.torch
        self._finger_joint_ids = finger_joint_ids.torch
        self._joint_pos_limits_t = self._robot.data.joint_pos_limits.torch.clone()

        tcp_body_ids, _ = self._robot.find_bodies(cfg.tcp_body_name)
        if len(tcp_body_ids) != 1:
            raise RuntimeError(f"Expected one TCP parent body named {cfg.tcp_body_name!r}, found {len(tcp_body_ids)}.")
        self._tcp_body_idx = tcp_body_ids[0]
        self._tcp_offset_pos = torch.tensor(cfg.tcp_offset_pos, device=dev).repeat(self.num_envs, 1)
        self._tcp_offset_quat = torch.tensor(cfg.tcp_offset_rot, device=dev).repeat(self.num_envs, 1)
        grasp_tcp_quat_c = torch.as_tensor(CUP_GRASP_TCP_QUAT_C, device=dev, dtype=torch.float32)
        self._desired_grasp_tcp_quat_c = math_utils.quat_unique(grasp_tcp_quat_c).repeat(self.num_envs, 1)

        # robot start pose (home) from the robot cfg, in ARM_JOINTS order
        joint_pos = cfg.scene.robot.init_state.joint_pos
        self._arm_home = torch.tensor([float(joint_pos[name]) for name in ARM_JOINTS], device=dev)
        self._gripper_open_position = float(joint_pos["panda_finger_joint.*"])
        self._gripper_open_width = 2.0 * self._gripper_open_position
        self._gripper_grasp_width = 2.0 * SOURCE_CUP_GEOMETRY.outer_half_extents[1]

        # nominal container and mixer poses (env frame) and the table height
        self.table_top = float(cfg.table_top)
        self._cup_reset_height = float(cfg.scene.source_cup.init_state.pos[2])
        self._cup_nominal_pos = torch.tensor(cfg.scene.source_cup.init_state.pos, device=dev, dtype=torch.float32)
        self._cup_nominal_quat = torch.tensor(cfg.scene.source_cup.init_state.rot, device=dev, dtype=torch.float32)
        self._mixer_nominal_pos = torch.tensor(cfg.scene.mixer.init_state.pos, device=dev, dtype=torch.float32)
        self._mixer_nominal_quat = torch.tensor(cfg.scene.mixer.init_state.rot, device=dev, dtype=torch.float32)

        # task geometry in the mixer frame
        def t(values):
            return torch.tensor(values, device=dev, dtype=torch.float32)

        self._chute_top_m = t(CHUTE_TOP_CENTRE)
        cx, cy, cz = CHUTE_TOP_CENTRE
        self._pour_target_m = t((cx, cy, cz + POUR_TARGET_ABOVE_CHUTE))
        self._bowl_lo_t, self._bowl_hi_t = t(MIXER_BOWL_REGION[0]), t(MIXER_BOWL_REGION[1])
        self._chute_lo_t, self._chute_hi_t = t(MIXER_CHUTE_REGION[0]), t(MIXER_CHUTE_REGION[1])
        self._mixer_lo_t, self._mixer_hi_t = t(MIXER_BBOX_REGION[0]), t(MIXER_BBOX_REGION[1])
        self._source_inner_lo_t = t(SOURCE_CUP_GEOMETRY.inner_bounds[0])
        self._source_inner_hi_t = t(SOURCE_CUP_GEOMETRY.inner_bounds[1])
        self._workspace_lower_t = t(cfg.workspace_lower_bound)
        self._workspace_upper_t = t(cfg.workspace_upper_bound)

        # beans: lattice in the container frame, shared by every reset
        self._num_beans = int(self._beans.num_bodies)
        local_positions = torch.tensor(beans_local_positions(cfg), device=dev, dtype=torch.float32)
        if local_positions.shape != (self._num_beans, 3):
            raise RuntimeError(
                f"Bean lattice has {local_positions.shape[0]} positions but the scene holds {self._num_beans} beans."
            )
        self._bean_local_position = local_positions
        self._bean_radius = bean_bounding_radius(cfg.bean_shape, cfg.bean_radius, cfg.bean_length)
        logger.info("Kitchen pour: %d beans per container (fill level %.2f).", self._num_beans, cfg.bean_fill_level)

        # episode state used by the (reused) pour-task terms
        self.pour_target_frac = torch.full((self.num_envs,), float(cfg.pour_target_frac), device=dev)
        self.episode_succeeded = torch.zeros(self.num_envs, device=dev, dtype=torch.bool)
        self._success_dwell_count = torch.zeros(self.num_envs, device=dev, dtype=torch.long)
        self._bean_region_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._bean_region_cache_step = -1
        self._bean_pos_e_cache: torch.Tensor | None = None
        self._bean_pos_e_cache_step = -1

    """
    Poses (env frame unless noted).
    """

    def _pose_w_to_e(self, pose_w: torch.Tensor) -> torch.Tensor:
        """World pose -> finite env-frame pose (env frames are translated copies of the world frame)."""
        pos = torch.nan_to_num(pose_w[:, :3], nan=0.0, posinf=0.0, neginf=0.0) - self.env_origins
        raw_quat = pose_w[:, 3:7]
        quat = torch.nan_to_num(raw_quat, nan=0.0, posinf=0.0, neginf=0.0)
        norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
        ident = torch.zeros_like(raw_quat)
        ident[:, 3] = 1.0
        valid = torch.isfinite(raw_quat).all(dim=-1, keepdim=True) & (norm > 1.0e-6)
        quat = torch.where(valid, quat / torch.clamp(norm, min=1.0e-6), ident)
        return torch.cat((pos, quat), dim=-1)

    def robot_root_pose_e(self) -> torch.Tensor:
        return self._pose_w_to_e(self._robot.data.root_link_pose_w.torch)

    def pose_e_to_b(self, pose_e: torch.Tensor) -> torch.Tensor:
        """Env-frame pose -> robot base (link0) frame."""
        root = self.robot_root_pose_e()
        pos, quat = math_utils.subtract_frame_transforms(root[:, :3], root[:, 3:7], pose_e[:, :3], pose_e[:, 3:7])
        return torch.cat((torch.nan_to_num(pos), torch.nan_to_num(quat)), dim=-1)

    def tcp_pose_e(self) -> torch.Tensor:
        """Tool-centre pose in the env frame (``(N, 7)``, position + xyzw quaternion)."""
        hand = self._pose_w_to_e(self._robot.data.body_link_pose_w.torch[:, self._tcp_body_idx])
        pos, quat = math_utils.combine_frame_transforms(
            hand[:, :3], hand[:, 3:7], self._tcp_offset_pos, self._tcp_offset_quat
        )
        return torch.cat((torch.nan_to_num(pos), torch.nan_to_num(quat)), dim=-1)

    def cup_pose_e(self) -> torch.Tensor:
        """Container pose (origin at the centre of its outer base)."""
        return self._pose_w_to_e(self._source_cup.data.root_link_pose_w.torch)

    def mixer_pose_e(self) -> torch.Tensor:
        """Mixer frame pose (origin at the top of the lid plate, +x = front)."""
        return self._pose_w_to_e(self._mixer.data.root_link_pose_w.torch)

    def _cup_point_e(self, height: float) -> torch.Tensor:
        pose = self.cup_pose_e()
        offset = torch.zeros((self.num_envs, 3), device=self.device)
        offset[:, 2] = height
        return pose[:, :3] + math_utils.quat_apply(pose[:, 3:7], offset)

    def cup_grasp_point_e(self) -> torch.Tensor:
        """Grasp point in the middle of the container walls."""
        return self._cup_point_e(CUP_GRASP_HEIGHT)

    def cup_mouth_e(self) -> torch.Tensor:
        """Centre of the container's opening."""
        return self._cup_point_e(SOURCE_CUP_GEOMETRY.rim_height)

    def _mixer_point_e(self, local: torch.Tensor) -> torch.Tensor:
        pose = self.mixer_pose_e()
        return pose[:, :3] + math_utils.quat_apply(pose[:, 3:7], local.expand(self.num_envs, -1))

    def chute_top_e(self) -> torch.Tensor:
        """Centre of the chute's top opening."""
        return self._mixer_point_e(self._chute_top_m)

    def pour_target_e(self) -> torch.Tensor:
        """Point above the chute that the container mouth is guided to."""
        return self._mixer_point_e(self._pour_target_m)

    def cup_tilt(self) -> torch.Tensor:
        """Angle between the container axis and the vertical [rad]."""
        axis = torch.zeros((self.num_envs, 3), device=self.device)
        axis[:, 2] = 1.0
        up = math_utils.quat_apply(self.cup_pose_e()[:, 3:7], axis)
        return torch.acos(torch.clamp(up[:, 2], -1.0, 1.0))

    def gripper_width(self) -> torch.Tensor:
        """Distance between the two fingers [m]."""
        finger_pos = self._robot.data.joint_pos.torch[:, self._finger_joint_ids]
        width = finger_pos.sum(dim=-1)
        valid = torch.isfinite(finger_pos).all(dim=-1) & torch.isfinite(width)
        return torch.where(valid, width, torch.full_like(width, self._gripper_open_width))

    """
    Beans.
    """

    def bean_pos_e(self) -> torch.Tensor:
        """Bean centres in the env frame, ``(N, B, 3)``."""
        step = int(getattr(self, "common_step_counter", -1))
        if self._bean_pos_e_cache is None or self._bean_pos_e_cache_step != step:
            self._bean_pos_e_cache = self._beans.data.body_link_pos_w.torch - self.env_origins[:, None, :]
            self._bean_pos_e_cache_step = step
        return self._bean_pos_e_cache

    def bean_vel_w(self) -> torch.Tensor:
        return self._beans.data.body_link_lin_vel_w.torch

    def _bean_region_masks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(in container, delivered into the mixer, spilled)`` bean masks, cached within one step.

        Delivered = in the chute (below its top opening) or in the bowl under the lid. Spilled = neither in the
        container nor delivered, and either down at table height (table, adapter, floor) or resting anywhere on
        the mixer (lid, head, base). Beans in the air are neither.
        """
        step = int(getattr(self, "common_step_counter", -1))
        if self._bean_region_cache is not None and self._bean_region_cache_step == step:
            return self._bean_region_cache
        points = self.bean_pos_e()
        margin = float(self.cfg.bean_count_margin)
        source = points_inside_box(points, self.cup_pose_e(), self._source_inner_lo_t, self._source_inner_hi_t, margin)
        mixer = self.mixer_pose_e()
        delivered = points_inside_box(points, mixer, self._bowl_lo_t, self._bowl_hi_t) | points_inside_box(
            points, mixer, self._chute_lo_t, self._chute_hi_t
        )
        delivered &= ~source
        on_mixer = points_inside_box(points, mixer, self._mixer_lo_t, self._mixer_hi_t)
        speed = torch.linalg.vector_norm(torch.nan_to_num(self.bean_vel_w()), dim=-1)
        resting = speed < float(self.cfg.spill_rest_speed)
        low = points[..., 2] <= self.table_top + float(self.cfg.spill_height_above_table)
        spilled = ~(source | delivered) & (low | (on_mixer & resting))
        self._bean_region_cache = (source, delivered, spilled)
        self._bean_region_cache_step = step
        return self._bean_region_cache

    def bean_fractions(self) -> torch.Tensor:
        """``(N, 3)``: fractions of beans in the container, delivered, spilled."""
        scale = max(self._num_beans, 1)
        return torch.stack([mask.sum(dim=1).float() / scale for mask in self._bean_region_masks()], dim=-1)

    def source_bean_fraction(self) -> torch.Tensor:
        return self.bean_fractions()[:, 0]

    """
    Reset.
    """

    def _yaw_noise_quat(self, n: int, max_yaw: float) -> torch.Tensor:
        yaw = (torch.rand(n, device=self.device) * 2.0 - 1.0) * float(max_yaw)
        axis = torch.zeros((n, 3), device=self.device)
        axis[:, 2] = 1.0
        return math_utils.quat_from_angle_axis(yaw, axis)

    def _noisy_pose(self, n, nominal_pos, nominal_quat, max_xy, max_yaw, env_ids) -> torch.Tensor:
        """Nominal pose + uniform horizontal / yaw noise, in the world frame."""
        pose = torch.zeros((n, 7), device=self.device)
        pose[:, :3] = nominal_pos
        pose[:, :2] += (torch.rand((n, 2), device=self.device) * 2.0 - 1.0) * float(max_xy)
        pose[:, :3] += self.env_origins[env_ids]
        pose[:, 3:7] = math_utils.quat_mul(self._yaw_noise_quat(n, max_yaw), nominal_quat.expand(n, -1))
        return pose

    def reset_kitchen_scene(self, env_ids: torch.Tensor | slice) -> None:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        n = env_ids.numel()
        if n == 0:
            return
        cfg = self.cfg

        # robot: home pose + noise (inside the joint limits), at rest, gripper open
        limits = self._joint_pos_limits_t[env_ids][:, self._arm_joint_ids]
        arm_q = self._arm_home.expand(n, -1) + (torch.rand((n, 7), device=self.device) * 2.0 - 1.0) * float(
            cfg.reset_joint_noise
        )
        arm_q = torch.minimum(torch.maximum(arm_q, limits[..., 0]), limits[..., 1])
        zeros7 = torch.zeros_like(arm_q)
        finger_q = torch.full((n, 2), self._gripper_open_position, device=self.device)
        zeros2 = torch.zeros_like(finger_q)
        robot = self._robot
        robot.write_joint_position_to_sim_index(position=arm_q, joint_ids=self._arm_joint_ids, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=zeros7, joint_ids=self._arm_joint_ids, env_ids=env_ids)
        robot.set_joint_position_target_index(target=arm_q, joint_ids=self._arm_joint_ids, env_ids=env_ids)
        robot.write_joint_position_to_sim_index(position=finger_q, joint_ids=self._finger_joint_ids, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=zeros2, joint_ids=self._finger_joint_ids, env_ids=env_ids)
        robot.set_joint_position_target_index(target=finger_q, joint_ids=self._finger_joint_ids, env_ids=env_ids)
        self.action_manager.get_term("gripper_action").set_reset_position(finger_q[:, :1], env_ids=env_ids)

        # container on the middle table, mixer on the left table (both with placement noise)
        cup_pose = self._noisy_pose(
            n, self._cup_nominal_pos, self._cup_nominal_quat, cfg.cup_pos_noise, cfg.cup_yaw_noise, env_ids
        )
        mixer_pose = self._noisy_pose(
            n, self._mixer_nominal_pos, self._mixer_nominal_quat, cfg.mixer_pos_noise, cfg.mixer_yaw_noise, env_ids
        )
        zeros6 = torch.zeros((n, 6), device=self.device)
        self._source_cup.write_root_pose_to_sim_index(root_pose=cup_pose, env_ids=env_ids)
        self._source_cup.write_root_velocity_to_sim_index(root_velocity=zeros6, env_ids=env_ids)
        self._mixer.write_root_pose_to_sim_index(root_pose=mixer_pose, env_ids=env_ids)
        self._mixer.write_root_velocity_to_sim_index(root_velocity=zeros6, env_ids=env_ids)

        self._reset_beans(env_ids, cup_pose)

        self._bean_region_cache = None
        self._bean_region_cache_step = -1
        self._bean_pos_e_cache = None
        self._bean_pos_e_cache_step = -1
        index_fill_(self.pour_target_frac, env_ids, float(cfg.pour_target_frac))
        index_fill_(self.episode_succeeded, env_ids, False)
        index_fill_(self._success_dwell_count, env_ids, 0)

    def _reset_beans(self, env_ids: torch.Tensor, cup_pose_w: torch.Tensor) -> None:
        """Bean lattice inside the (restored) container, at rest (same scheme as the pour env)."""
        n = env_ids.shape[0]
        local = self._bean_local_position[None].expand(n, -1, -1).clone()
        jitter = float(self.cfg.bean_reset_jitter)
        if jitter > 0.0:
            local[..., :2] += (torch.rand_like(local[..., :2]) * 2.0 - 1.0) * jitter
        cup_quat = cup_pose_w[:, None, 3:7].expand(-1, self._num_beans, -1)
        bean_pos = cup_pose_w[:, None, :3] + math_utils.quat_apply(cup_quat, local)
        if self.cfg.bean_shape == "sphere":
            bean_quat = cup_quat
        else:
            yaw = torch.rand((n, self._num_beans), device=self.device) * (2.0 * torch.pi)
            axis = torch.zeros((n, self._num_beans, 3), device=self.device)
            axis[..., 2] = 1.0
            bean_quat = math_utils.quat_mul(cup_quat, math_utils.quat_from_angle_axis(yaw, axis))
        self._beans.write_body_pose_to_sim_index(body_poses=torch.cat((bean_pos, bean_quat), dim=-1), env_ids=env_ids)
        self._beans.write_body_velocity_to_sim_index(
            body_velocities=torch.zeros((n, self._num_beans, 6), device=self.device), env_ids=env_ids
        )
