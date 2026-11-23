# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitchen beans-pour task: the Franka of the pour task takes the bean container from the middle table of the
Hestia kitchen (scenes/u_table_layout) and pours the beans into the chute of the mixer on the left table (PhysX).

The environment lives in :mod:`kitchen_env` / :mod:`kitchen_env_cfg`, the kitchen geometry in :mod:`layout`, and the
gym registration in :mod:`config.franka`.
"""
