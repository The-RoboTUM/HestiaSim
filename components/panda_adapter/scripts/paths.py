"""Where the scripts write their results.

Two layouts are supported:
    aluprofil_table/            (dev folder)   scripts and results side by side
    <component>/scripts/        (HestiaSimulation repo)   results go one level up, into <component>/
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
IN_SCRIPTS_DIR = os.path.basename(HERE) == "scripts"
ROOT = os.path.dirname(HERE) if IN_SCRIPTS_DIR else HERE

MESHES = os.path.join(ROOT, "meshes")  # meter-based OBJ for MuJoCo / Isaac
DRAWINGS = os.path.join(ROOT, "drawings")
STL_OUT = ROOT if IN_SCRIPTS_DIR else os.path.join(ROOT, "exports")  # mm STL for the hardware team
USD = os.path.join(ROOT, "usd")
