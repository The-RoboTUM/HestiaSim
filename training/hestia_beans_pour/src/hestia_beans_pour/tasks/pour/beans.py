# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative configuration of the rigid beans that fill the source cup.

Each bean is an individual PhysX rigid body. All beans of one environment are grouped in a single
:class:`~isaaclab.assets.RigidObjectCollection` so their state is exposed as ``(num_envs, num_beans, ...)``
tensors. Beans start on a collision-free lattice inside the source-cup cavity; the lattice is expressed in
the source-cup frame so that every reset row (including tilted or lifted cups) receives the same fill.
"""

from __future__ import annotations

import math
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sim.schemas import MassCfg, UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import clone
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxRigidBodyCfg

from .geometry import CupGeometry

BeanShape = Literal["sphere", "capsule"]

BEAN_PRIM_NAME = "Bean_{:03d}"
"""Name pattern of one bean prim directly below the environment root."""


def bean_bounding_radius(shape: BeanShape, radius: float, length: float) -> float:
    """Return the radius of the sphere that encloses one bean [m]."""
    if shape == "sphere":
        return float(radius)
    if shape == "capsule":
        return float(radius) + 0.5 * float(length)
    raise ValueError(f"Unsupported bean shape {shape!r}; expected 'sphere' or 'capsule'.")


def bean_lattice_c(
    geometry: CupGeometry,
    *,
    bounding_radius: float,
    gap: float,
    fill_level: float,
    max_beans: int | None = None,
) -> list[tuple[float, float, float]]:
    """Return collision-free bean centers inside a cup cavity, expressed in the cup frame [m].

    Beans are placed on a cubic lattice with ``2 * bounding_radius + gap`` spacing, centered horizontally,
    starting at the cavity floor and stacked up to ``fill_level`` of the cavity height. Every bean keeps at
    least ``gap`` clearance to the walls, the floor, and its neighbours, so no contact exists at reset.

    Args:
        geometry: Source-cup dimensions.
        bounding_radius: Radius of a sphere enclosing one bean [m].
        gap: Minimum clearance between neighbouring beans and between beans and walls [m].
        fill_level: Fraction of the cavity height that may be filled, in ``(0, 1]``.
        max_beans: Optional cap on the number of beans, filled bottom-up.

    Returns:
        Lattice positions ordered bottom layer first.
    """
    level = float(fill_level)
    if not math.isfinite(level) or not 0.0 < level <= 1.0:
        raise ValueError(f"bean fill_level must lie in (0, 1], got {fill_level}.")
    radius = float(bounding_radius)
    gap = float(gap)
    if radius <= 0.0 or gap < 0.0:
        raise ValueError("bean bounding radius must be positive and gap nonnegative.")
    spacing = 2.0 * radius + gap
    clearance = radius + gap

    def axis_centers(size: float) -> list[float]:
        usable = float(size) - 2.0 * clearance
        if usable < 0.0:
            raise ValueError(f"Beans with bounding radius {radius} m do not fit in a {size} m wide cavity.")
        count = math.floor(usable / spacing + 1.0e-9) + 1
        first = -0.5 * (count - 1) * spacing
        return [first + index * spacing for index in range(count)]

    xs = axis_centers(geometry.inner_width)
    ys = axis_centers(geometry.inner_depth)
    fill_height = level * geometry.cavity_depth
    layers = max(1, math.floor((fill_height - 2.0 * clearance) / spacing + 1.0e-9) + 1)
    max_layers = max(1, math.floor((geometry.cavity_depth - 2.0 * clearance) / spacing + 1.0e-9) + 1)
    layers = min(layers, max_layers)
    z0 = geometry.bottom_thickness + clearance

    positions = [(x, y, z0 + layer * spacing) for layer in range(layers) for y in ys for x in xs]
    if max_beans is not None:
        if max_beans <= 0:
            raise ValueError("max_beans must be positive or None.")
        if max_beans > len(positions):
            raise ValueError(
                f"Requested {max_beans} beans but only {len(positions)} fit at fill level {fill_level}; "
                "raise the fill level or reduce the bean size."
            )
        positions = positions[:max_beans]
    return positions


def build_beans_collection_cfg(
    local_positions: list[tuple[float, float, float]],
    *,
    cup_pos: tuple[float, float, float],
    shape: BeanShape,
    radius: float,
    length: float,
    mass: float,
    friction: float,
    restitution: float,
    color: tuple[float, float, float],
) -> RigidObjectCollectionCfg:
    """Build one rigid-body configuration per bean, grouped as a collection.

    Args:
        local_positions: Bean centers in the source-cup frame [m].
        cup_pos: Default source-cup position used for the authored initial state [m].
        shape: Bean primitive, either a sphere or a capsule.
        radius: Sphere or capsule radius [m].
        length: Capsule cylinder length [m]; ignored for spheres.
        mass: Mass of one bean [kg].
        friction: Static and dynamic friction coefficient.
        restitution: Restitution coefficient.
        color: Diffuse RGB color in ``[0, 1]``.
    """
    common = dict(
        rigid_props=[
            UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=False),
            PhysxRigidBodyCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=0.5,
                max_linear_velocity=10.0,
                max_angular_velocity=500.0,
                disable_gravity=False,
            ),
        ],
        mass_props=MassCfg(mass=float(mass)),
        collision_props=[
            UsdPhysicsCollisionCfg(collision_enabled=True),
            PhysxCollisionCfg(contact_offset=0.002, rest_offset=0.0),
        ],
        physics_material=RigidBodyMaterialBaseCfg(
            static_friction=float(friction),
            dynamic_friction=float(friction),
            restitution=float(restitution),
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color), roughness=0.6),
    )
    if shape == "sphere":
        spawn = sim_utils.SphereCfg(radius=float(radius), **common)
    elif shape == "capsule":
        spawn = sim_utils.CapsuleCfg(radius=float(radius), height=float(length), axis="X", **common)
    else:
        raise ValueError(f"Unsupported bean shape {shape!r}; expected 'sphere' or 'capsule'.")

    rigid_objects = {}
    for index, local in enumerate(local_positions):
        name = BEAN_PRIM_NAME.format(index)
        rigid_objects[name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=clone(spawn),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(float(c + o) for c, o in zip(cup_pos, local, strict=True)),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)
