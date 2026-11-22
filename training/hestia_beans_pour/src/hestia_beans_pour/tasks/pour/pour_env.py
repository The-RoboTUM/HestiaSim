# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL environment for a Franka pouring rigid beans between two cups (Isaac Sim PhysX).

The dynamic source cup and kinematic receiver are loaded from the Franka Pour cups USD, with their
authored colliders replaced by hollow five-box PhysX colliders so the beans can rest inside them. Each
bean is an independent rigid body; all beans of one environment form one rigid-object collection.

Resets restore the robot and both cups from rows of the published Franka Pour reset dataset and refill
the source cup with a fixed bean lattice expressed in the restored cup frame. The policy commands filtered
relative arm joint-position deltas and a filtered binary symmetric finger command.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import index_fill_
from isaaclab.utils import math as math_utils
from isaaclab.utils.assets import retrieve_file_path

from .beans import bean_bounding_radius
from .geometry import (
    CUP_GRASP_HEIGHT,
    CUP_GRASP_TCP_QUAT_C,
    SOURCE_CUP_GEOMETRY,
    TARGET_CUP_GEOMETRY,
    points_inside_box,
)
from .pour_env_cfg import beans_local_positions, configure_beans, reset_dataset_task_contract
from .reset_dataset_io import RESET_DATASET_STATE_NAMES, reset_dataset_validate_runtime

if TYPE_CHECKING:
    from isaaclab.assets import RigidObjectCollection

    from .pour_env_cfg import FrankaBeansPourEnvCfg

logger = logging.getLogger(__name__)

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
GRASPING_CATEGORY = 1


def _resolve_reset_dataset_path(configured_path: str) -> Path:
    """Resolve a local dataset or download a remote dataset through the standard asset cache."""
    if "://" in configured_path:
        try:
            return Path(retrieve_file_path(configured_path)).resolve()
        except (FileNotFoundError, RuntimeError) as exc:
            raise FileNotFoundError(
                f"Franka Pour reset dataset unavailable: {configured_path}. Generate a local artifact from the "
                "Isaac Lab repository root with: uv run python scripts/tools/generate_franka_pour_reset_dataset.py "
                "--device cuda:0, then pass env.reset_dataset_path=<absolute path to reset_dataset.pt>."
            ) from exc
    path = Path(configured_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Franka Pour reset dataset not found: {path}.")
    return path


class FrankaBeansPourEnv(ManagerBasedRLEnv):
    """Franka grasping a dynamic cup of rigid beans and pouring them into a receiver."""

    cfg: FrankaBeansPourEnvCfg

    def __init__(self, cfg: FrankaBeansPourEnvCfg, render_mode: str | None = None, **kwargs):
        # Bean count and size are configurable from the command line, so rebuild the collection here.
        configure_beans(cfg)
        super().__init__(cfg, render_mode, **kwargs)

    def load_managers(self) -> None:
        self._setup_after_physics()
        super().load_managers()

    """
    Setup.
    """

    def _setup_after_physics(self) -> None:
        """Bind scene assets, cache task constants, and load the reset dataset."""
        dev = self.device
        self._robot = self.scene["robot"]
        self._source_cup = self.scene["source_cup"]
        self._target_cup = self.scene["target_cup"]
        self._beans: RigidObjectCollection = self.scene["beans"]

        arm_joint_ids, _ = self._robot.find_joints(ARM_JOINTS, preserve_order=True, as_proxy=True)
        finger_joint_ids, _ = self._robot.find_joints(FINGER_JOINTS, preserve_order=True, as_proxy=True)
        self._arm_joint_ids = arm_joint_ids.torch
        self._finger_joint_ids = finger_joint_ids.torch
        self._joint_pos_limits_t = self._robot.data.joint_pos_limits.torch.clone()
        self.env_origins = self.scene.env_origins.to(device=dev, dtype=torch.float32)

        tcp_body_ids, _ = self._robot.find_bodies(self.cfg.tcp_body_name)
        if len(tcp_body_ids) != 1:
            raise RuntimeError(
                f"Expected one TCP parent body named {self.cfg.tcp_body_name!r}, found {len(tcp_body_ids)}."
            )
        self._tcp_body_idx = tcp_body_ids[0]
        self._tcp_offset_pos = torch.tensor(self.cfg.tcp_offset_pos, device=dev).repeat(self.num_envs, 1)
        self._tcp_offset_quat = torch.tensor(self.cfg.tcp_offset_rot, device=dev).repeat(self.num_envs, 1)
        grasp_tcp_quat_c = torch.as_tensor(CUP_GRASP_TCP_QUAT_C, device=dev, dtype=torch.float32)
        self._desired_grasp_tcp_quat_c = math_utils.quat_unique(grasp_tcp_quat_c).repeat(self.num_envs, 1)
        self._gripper_open_width = 2.0 * float(self.cfg.scene.robot.init_state.joint_pos["panda_finger_joint.*"])
        self._gripper_grasp_width = 2.0 * SOURCE_CUP_GEOMETRY.outer_half_extents[1]
        self._cup_reset_height = float(self.cfg.scene.source_cup.init_state.pos[2])
        self._source_outer_half_t = torch.tensor(SOURCE_CUP_GEOMETRY.outer_half_extents, device=dev)
        self._target_outer_half_t = torch.tensor(TARGET_CUP_GEOMETRY.outer_half_extents, device=dev)
        self._source_inner_lo_t = torch.as_tensor(SOURCE_CUP_GEOMETRY.inner_bounds[0], device=dev)
        self._source_inner_hi_t = torch.as_tensor(SOURCE_CUP_GEOMETRY.inner_bounds[1], device=dev)
        self._target_inner_lo_t = torch.as_tensor(TARGET_CUP_GEOMETRY.inner_bounds[0], device=dev)
        self._target_inner_hi_t = torch.as_tensor(TARGET_CUP_GEOMETRY.inner_bounds[1], device=dev)
        self._workspace_lower_t = torch.as_tensor(self.cfg.workspace_lower_bound, device=dev, dtype=torch.float32)
        self._workspace_upper_t = torch.as_tensor(self.cfg.workspace_upper_bound, device=dev, dtype=torch.float32)

        # Bean lattice in the source-cup frame, shared by every reset row.
        self._num_beans = int(self._beans.num_bodies)
        local_positions = torch.tensor(beans_local_positions(self.cfg), device=dev, dtype=torch.float32)
        if local_positions.shape != (self._num_beans, 3):
            raise RuntimeError(
                f"Bean lattice has {local_positions.shape[0]} positions but the scene holds {self._num_beans} beans."
            )
        self._bean_local_position = local_positions
        self._bean_radius = bean_bounding_radius(self.cfg.bean_shape, self.cfg.bean_radius, self.cfg.bean_length)

        self._load_reset_dataset(self.cfg.reset_dataset_path)

        self.reset_dataset_row_id = torch.full((self.num_envs,), -1, device=dev, dtype=torch.long)
        self.pour_target_frac = torch.full((self.num_envs,), float(self.cfg.pour_target_frac), device=dev)
        self.episode_succeeded = torch.zeros(self.num_envs, device=dev, dtype=torch.bool)
        self._success_dwell_count = torch.zeros(self.num_envs, device=dev, dtype=torch.long)
        self._lost_grasp_dwell_count = torch.zeros(self.num_envs, device=dev, dtype=torch.long)
        self._lifted_grasp_seen = torch.zeros(self.num_envs, device=dev, dtype=torch.bool)
        self._bean_region_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._bean_region_cache_step = -1
        self._bean_pos_e_cache: torch.Tensor | None = None
        self._bean_pos_e_cache_step = -1

    def _load_reset_dataset(self, configured_path: str) -> None:
        """Load and stage a validated rigid-state dataset on the simulation device."""
        cache_path = _resolve_reset_dataset_path(configured_path)
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        _, states, _ = reset_dataset_validate_runtime(
            payload,
            expected_content_sha256=self.cfg.reset_dataset_content_sha256,
            expected_task_contract=reset_dataset_task_contract(self.cfg),
        )
        self._reset_dataset_states = {
            name: states[name].to(device=self.device, non_blocking=True) for name in RESET_DATASET_STATE_NAMES
        }
        logger.info(
            "Loaded reset dataset %s (%s) with %d rows.",
            cache_path,
            payload["content_sha256"],
            len(states["category"]),
        )

    """
    Task state.
    """

    def _pose_w_to_e(self, pose_w: torch.Tensor) -> torch.Tensor:
        """Convert a world-frame pose to a finite environment-frame pose."""
        pos = torch.nan_to_num(pose_w[:, :3], nan=0.0, posinf=0.0, neginf=0.0) - self.env_origins
        raw_quat = pose_w[:, 3:7]
        quat = torch.nan_to_num(raw_quat, nan=0.0, posinf=0.0, neginf=0.0)
        norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
        ident = torch.zeros_like(raw_quat)
        ident[:, 3] = 1.0
        valid = torch.isfinite(raw_quat).all(dim=-1, keepdim=True) & (norm > 1.0e-6)
        quat = torch.where(valid, quat / torch.clamp(norm, min=1.0e-6), ident)
        return torch.cat((pos, quat), dim=-1)

    def cup_pose_e(self) -> torch.Tensor:
        """Source-cup pose in the env frame: ``(num_envs, 7)`` pos + xyzw quat."""
        return self._pose_w_to_e(self._source_cup.data.root_link_pose_w.torch)

    def target_pose_e(self) -> torch.Tensor:
        """Receiver pose in the env frame: ``(num_envs, 7)`` pos + xyzw quat."""
        return self._pose_w_to_e(self._target_cup.data.root_link_pose_w.torch)

    def tcp_pose_e(self) -> torch.Tensor:
        """Tool-centre pose in the robot-root/environment frame."""
        body_pose_w = self._robot.data.body_link_pose_w.torch[:, self._tcp_body_idx]
        root_pose_w = self._robot.data.root_link_pose_w.torch
        pos, quat = math_utils.subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], body_pose_w[:, :3], body_pose_w[:, 3:7]
        )
        pos, quat = math_utils.combine_frame_transforms(pos, quat, self._tcp_offset_pos, self._tcp_offset_quat)
        return torch.cat((torch.nan_to_num(pos), torch.nan_to_num(quat)), dim=-1)

    def cup_grasp_point_e(self) -> torch.Tensor:
        """Grasp point at the middle of the source-cup walls, in env coordinates."""
        pose = self.cup_pose_e()
        offset = torch.zeros((self.num_envs, 3), device=self.device)
        offset[:, 2] = CUP_GRASP_HEIGHT
        return pose[:, :3] + math_utils.quat_apply(pose[:, 3:7], offset)

    def gripper_width(self) -> torch.Tensor:
        """Distance represented by the two symmetric Panda finger joint positions [m]."""
        finger_pos = self._robot.data.joint_pos.torch[:, self._finger_joint_ids]
        width = finger_pos.sum(dim=-1)
        valid = torch.isfinite(finger_pos).all(dim=-1) & torch.isfinite(width)
        return torch.where(valid, width, torch.full_like(width, self._gripper_open_width))

    def bean_pos_e(self) -> torch.Tensor:
        """Per-env bean center positions in env coordinates, shape ``(N, B, 3)``."""
        step = int(getattr(self, "common_step_counter", -1))
        if self._bean_pos_e_cache is None or self._bean_pos_e_cache_step != step:
            self._bean_pos_e_cache = self._beans.data.body_link_pos_w.torch - self.env_origins[:, None, :]
            self._bean_pos_e_cache_step = step
        return self._bean_pos_e_cache

    def bean_vel_w(self) -> torch.Tensor:
        """Per-env bean linear velocities, shape ``(N, B, 3)``."""
        return self._beans.data.body_link_lin_vel_w.torch

    def _bean_region_masks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Boolean ``(source, target, spilled)`` bean masks, cached within one manager step."""
        step = int(getattr(self, "common_step_counter", -1))
        if self._bean_region_cache is not None and self._bean_region_cache_step == step:
            return self._bean_region_cache
        points = self.bean_pos_e()
        margin = float(self.cfg.bean_count_margin)
        source = points_inside_box(points, self.cup_pose_e(), self._source_inner_lo_t, self._source_inner_hi_t, margin)
        target_region = points_inside_box(
            points, self.target_pose_e(), self._target_inner_lo_t, self._target_inner_hi_t, margin
        )
        # Geometric overlap is not delivery: beans still held by a nested source cup do not count.
        target = target_region & ~source
        # A bean resting on the table has its center one bounding radius above the table top.
        spill_height = float(self.cfg.spill_table_height) + self._bean_radius + margin
        spilled = ~(source | target) & (points[..., 2] <= spill_height)
        self._bean_region_cache = (source, target, spilled)
        self._bean_region_cache_step = step
        return source, target, spilled

    """
    Reset.
    """

    def _reset_from_dataset(self, env_ids: torch.Tensor) -> None:
        """Restore exact dataset rows for the robot and cups, then refill the source cup with beans."""
        rows = self.reset_dataset_row_id[env_ids]
        states = self._reset_dataset_states
        arm_q = states["arm_joint_position"][rows]
        arm_qd = states["arm_joint_velocity"][rows]
        finger_q = states["finger_joint_position"][rows]
        finger_qd = states["finger_joint_velocity"][rows]
        finger_target = states["finger_joint_target"][rows]

        self._robot.write_joint_position_to_sim_index(position=arm_q, joint_ids=self._arm_joint_ids, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=arm_qd, joint_ids=self._arm_joint_ids, env_ids=env_ids)
        self._robot.set_joint_position_target_index(target=arm_q, joint_ids=self._arm_joint_ids, env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(
            position=finger_q, joint_ids=self._finger_joint_ids, env_ids=env_ids
        )
        self._robot.write_joint_velocity_to_sim_index(
            velocity=finger_qd, joint_ids=self._finger_joint_ids, env_ids=env_ids
        )
        self._robot.set_joint_position_target_index(
            target=finger_target, joint_ids=self._finger_joint_ids, env_ids=env_ids
        )
        self.action_manager.get_term("gripper_action").set_reset_position(finger_target[:, :1], env_ids=env_ids)

        source_pose = states["source_root_pose"][rows].clone()
        source_pose[:, :3] += self.env_origins[env_ids]
        source_velocity = states["source_root_velocity"][rows]
        target_pose = states["target_root_pose"][rows].clone()
        target_pose[:, :3] += self.env_origins[env_ids]
        self._source_cup.write_root_pose_to_sim_index(root_pose=source_pose, env_ids=env_ids)
        self._source_cup.write_root_velocity_to_sim_index(root_velocity=source_velocity, env_ids=env_ids)
        self._target_cup.write_root_pose_to_sim_index(root_pose=target_pose, env_ids=env_ids)
        self._target_cup.write_root_velocity_to_sim_index(
            root_velocity=states["target_root_velocity"][rows], env_ids=env_ids
        )

        self._reset_beans(env_ids, source_pose, source_velocity)

        self._bean_region_cache = None
        self._bean_region_cache_step = -1
        self._bean_pos_e_cache = None
        self._bean_pos_e_cache_step = -1
        index_fill_(self.episode_succeeded, env_ids, False)
        index_fill_(self._success_dwell_count, env_ids, 0)
        index_fill_(self._lost_grasp_dwell_count, env_ids, 0)
        # Seed the dropped-grasp latch for validated grasp rows; non-grasp rows start clear.
        self._lifted_grasp_seen[env_ids] = states["category"][rows] == GRASPING_CATEGORY

    def _reset_beans(self, env_ids: torch.Tensor, source_pose_w: torch.Tensor, source_velocity_w: torch.Tensor):
        """Place the bean lattice inside the restored source cup, moving rigidly with it."""
        num_resets = env_ids.shape[0]
        local = self._bean_local_position[None].expand(num_resets, -1, -1).clone()
        jitter = float(self.cfg.bean_reset_jitter)
        if jitter > 0.0:
            local[..., :2] += (torch.rand_like(local[..., :2]) * 2.0 - 1.0) * jitter
        cup_quat = source_pose_w[:, None, 3:7].expand(-1, self._num_beans, -1)
        offset_w = math_utils.quat_apply(cup_quat, local)
        bean_pos = source_pose_w[:, None, :3] + offset_w
        if self.cfg.bean_shape == "sphere":
            bean_quat = cup_quat
        else:
            # Random yaw (in the cup frame) so elongated beans do not all point the same way.
            yaw = torch.rand((num_resets, self._num_beans), device=self.device) * (2.0 * torch.pi)
            axis = torch.zeros((num_resets, self._num_beans, 3), device=self.device)
            axis[..., 2] = 1.0
            bean_quat = math_utils.quat_mul(cup_quat, math_utils.quat_from_angle_axis(yaw, axis))
        bean_pose = torch.cat((bean_pos, bean_quat), dim=-1)
        # Rigid-body velocity of the cup at each bean center.
        lin = source_velocity_w[:, None, :3] + torch.linalg.cross(
            source_velocity_w[:, None, 3:].expand_as(offset_w), offset_w, dim=-1
        )
        ang = source_velocity_w[:, None, 3:].expand_as(offset_w)
        bean_velocity = torch.cat((lin, ang), dim=-1)
        self._beans.write_body_pose_to_sim_index(body_poses=bean_pose, env_ids=env_ids)
        self._beans.write_body_velocity_to_sim_index(body_velocities=bean_velocity, env_ids=env_ids)

    def reset_pour_scene(self, env_ids: torch.Tensor | slice) -> None:
        """Restore selected environments from reset-dataset rows."""
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if len(env_ids) == 0:
            return
        self._reset_from_dataset(env_ids.to(device=self.device, dtype=torch.long))
