# Container

Container 01 with one handle.

| Version | File | Made by | What changed |
|---|---|---|---|
| v2 | `container01_v2.stl` (mm, 79 MB, 1.6M triangles) | maker unknown; received from Niklas on 2026-11-23 as `container01_v2.stl` | full-detail original. Received in metres at an assembly offset (z ≈ 1000.9 m); scaled x1000 to mm and moved so the bottom centre is at the origin; shape unchanged |
| **v3 (current)** | `container01_v3.stl` (mm, 2.5 MB, 50k triangles) | simplified from v2 (2026-11-23) | **simplified copy of v2 for simulation**: same shape, 32x smaller file. Accurate enough for simulation: its vertices lie within 0.03 mm of v2's surface, same size and origin, volume within 0.01 %. No holes; 4 of 75k edges are shared by more than two triangles, so it is not strictly "watertight" (irrelevant for visuals and collision hulls) |

- Origin and front: centre of the bottom face; +x = handle side (handle 20 mm wide, sticks out 30 mm)
- Size: 190 x 120 x 70 mm (body without handle 160 x 120 mm)

Use v3 in scenes and training; keep v2 as the full-detail original (e.g. for the hardware team).
