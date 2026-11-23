# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitchen pour geometry (pure Python: runs without Isaac Sim, loads layout.py by path)."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.kitless]

LAYOUT_PY = Path(__file__).resolve().parents[1] / "src" / "hestia_beans_pour" / "tasks" / "kitchen_pour" / "layout.py"


@pytest.fixture(scope="module")
def layout():
    spec = importlib.util.spec_from_file_location("kitchen_pour_layout", LAYOUT_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_poses_match_the_u_scene(layout):
    k = layout.kitchen_layout()
    # values of scenes/u_table_layout/README.md
    assert k.robot_pos == pytest.approx((1.181, 0.244, 0.69), abs=1e-3)
    assert k.robot_yaw_deg == pytest.approx(135.0)
    assert k.robot_home == pytest.approx((0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853))
    assert k.mixer_pos[:2] == pytest.approx((1.2525, 0.9), abs=1e-4)
    assert k.mixer_pos[2] == pytest.approx(0.68 + 0.233)
    assert k.mixer_yaw_deg == pytest.approx(180.0)


def test_container_on_the_middle_table(layout):
    k = layout.kitchen_layout()
    assert k.cup_pos == pytest.approx((0.75, 0.28, 0.68))
    # inside the bottom table's top (x in +-0.9, top y from -0.36 to 0.48), with the container's 28 mm half width
    assert -0.9 + 0.028 < k.cup_pos[0] < 0.9 - 0.028
    assert -0.36 + 0.028 < k.cup_pos[1] < 0.48 - 0.028
    assert "15.0 cm" in layout.placement_report() and "20.0 cm" in layout.placement_report()


def test_quaternions_are_xyzw(layout):
    assert layout.quat_xyzw_from_yaw(0.0) == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert layout.quat_xyzw_from_yaw(180.0) == pytest.approx((0.0, 0.0, 1.0, 0.0), abs=1e-12)


def test_chute_hole_is_open(layout):
    """No collider box may cover the chute's floor opening (else the beans could never enter the mixer)."""
    x0, x1 = layout.CHUTE_HOLE_X
    hy = layout.CHUTE_HOLE_HALF_Y
    probes = [(x, y, -0.005) for x in (x0 + 0.004, 0.5 * (x0 + x1), x1 - 0.004) for y in (-hy + 0.004, 0.0, hy - 0.004)]
    for name, centre, half, pitch in layout.MIXER_COLLIDER_BOXES:
        c, s = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
        for px, py, pz in probes:
            dx, dy, dz = px - centre[0], py - centre[1], pz - centre[2]
            # into the box frame (inverse rotation about y)
            lx, lz = c * dx - s * dz, s * dx + c * dz
            inside = abs(lx) <= half[0] and abs(dy) <= half[1] and abs(lz) <= half[2]
            assert not inside, f"collider {name} blocks the chute hole at {(px, py, pz)}"


def test_pour_target_above_the_chute(layout):
    k = layout.kitchen_layout()
    target = k.pour_target
    hole = k.mixer_point((0.5 * sum(layout.CHUTE_HOLE_X), 0.0, 0.0))
    assert math.hypot(target[0] - hole[0], target[1] - hole[1]) < 0.01
    assert target[2] - hole[2] == pytest.approx(layout.CHUTE_TOP_CENTRE[2] + layout.POUR_TARGET_ABOVE_CHUTE)


def test_statics_usd(layout, tmp_path):
    path = Path(layout.write_kitchen_statics_usda(tmp_path))
    text = path.read_text()
    assert text.count('def Cube "') == 3 * 13
    assert 'def Cylinder "PandaAdapter"' in text
    # writing again reuses the same content-addressed file
    assert layout.write_kitchen_statics_usda(tmp_path) == str(path)
