# Container

Container 01 with one handle.

| Version | File | Made by | What changed |
|---|---|---|---|
| v2 | `container01_v2.stl` (mm, 79 MB, 1.6M triangles) | maker unknown; received from Niklas on 2026-11-23 as `container01_v2.stl` | full-detail original. Received in metres at an assembly offset (z ≈ 1000.9 m); scaled x1000 to mm and moved so the bottom centre is at the origin; shape unchanged |
| **v3 (current)** | `container01_v3.stl` (mm, 2.5 MB, 50k triangles) | simplified from v2 (2026-11-23) | **simplified copy of v2 for simulation**: same shape, 32x smaller file. Accurate enough for simulation: its vertices lie within 0.03 mm of v2's surface, same size and origin, volume within 0.01 %. No holes; 4 of 75k edges are shared by more than two triangles, so it is not strictly "watertight" (irrelevant for visuals and collision hulls) |

- Origin and front: centre of the bottom face; +x = handle side (handle 20 mm wide, sticks out 30 mm)
- Size: 190 x 120 x 70 mm (body without handle 160 x 120 mm)

Use v3 in scenes and training; keep v2 as the full-detail original (e.g. for the hardware team).

## How v3 was made (simplifying a heavy mesh)

`scripts/simplify_mesh.py` reduces the number of triangles and keeps units, origin and shape; it then
prints an accuracy check. Running it again gives exactly the committed v3 (checked byte for byte).

```bash
cd components/container/scripts
uv run --no-project --with trimesh --with numpy --with scipy --with rtree --with fast-simplification \
    python simplify_mesh.py ../container01_v2.stl ../container01_v3.stl --faces 50000
```

1. **Simplify:** quadric edge collapse (`trimesh` + `fast-simplification`) merges triangles where the surface
   is flat, so flat walls lose almost all triangles and edges and the handle keep their shape.
2. **Check accuracy and physics** (printed by the script). For v3:
   - size unchanged (190 x 120 x 69.95 mm); vertices at most 0.024 mm from v2's surface (exact, 20k vertices)
   - volume 99.998 % of v2, so the **same mass** for the same material
   - **centre of mass** shifted by 0.0004 mm (at x -7.6, y 0, z 26.6 mm); **inertia** within 0.004 %
   - no holes; 4 edges shared by more than two triangles (fine for simulation)

   Same size, mass, centre of mass and inertia means the object moves the same when the robot lifts or tips it.
3. **Choose the triangle count:** 50k, 100k and 150k were all equally accurate (this shape is mostly flat),
   so the smallest, 50k (2.5 MB), was taken. For another object, try a few counts and take the smallest one
   whose check still looks like this.

Works for any STL; use it when a received mesh is too heavy for simulation.

## Shape (measured from v2, 2026-11-23)

Container frame: mm, origin = centre of the bottom face, +x = handle side, Z-up.

```
 side view (y = 0)                       top view
 z                                       y
 70 ┌───────────────────┬─────┬───┐      60 ┌───────────────────┐
    │ wall 5        open│ top bar │         │ ┌───────────────┐ │    ┌───┐ 10
    │               top │z 55-70  │         │ │   inside      │ ├────┤   │ handle 20 wide
    │                   ├─────┤   │         │ │ 150 x 110     │ ├────┤   │
    │      inside       │ gap │out│         │ └───────────────┘ │    └───┘-10
    │                   │15 mm│bar│     -60 └───────────────────┘
  5 │___________________│(open│   │        -95                 65  80  95  x
  0 └───────────────────┘below)└──┘ z 10
  -95                  65    80   95  x
```

| Part | x | y | z | Notes |
|---|---|---|---|---|
| Body (outside) | -95 .. 65 | -60 .. 60 | 0 .. 69.95 | open top, no lid |
| Walls | 5 mm thick | | 5 .. 69.95 | straight (no taper): inside x -90 .. 60, y -55 .. 55 |
| Bottom | | | 0 .. 5 | 5 mm thick |
| Handle top bar | 65 .. 80 | -10 .. 10 | 55 .. 69.95 | joins the body's +x wall at the top |
| Handle outer bar | 80 .. 95 | -10 .. 10 | 10 .. 69.95 | bottom is rounded/sloped: z 17.5 at x 82, 11.7 at x 88, 10 at x 94 |
| Handle gap | 65 .. 80 (15 mm) | -10 .. 10 | 0 .. 55 | between body wall and outer bar, **open at the bottom** (a hook, not a closed loop) |

Volume of material 292 cm³ (22 % of the outer box: thin walls). Mass and friction: not known yet; mass =
292 cm³ x density once the material is known (e.g. PLA, 1.24 g/cm³, would give about 360 g).

## Collision boxes (for Isaac / MuJoCo)

A single convex hull (MuJoCo's default for meshes) would fill the inside and the handle gap, so beans could
not sit inside and fingers could not enter the gap. Use these 7 boxes as the collision shape and v3 only as
the visual mesh. Centre and full size in mm, container frame:

| Box | Centre (x, y, z) | Size (x, y, z) |
|---|---|---|
| bottom | (-15, 0, 2.5) | 160 x 120 x 5 |
| wall -x | (-92.5, 0, 37.475) | 5 x 110 x 64.95 |
| wall +x (handle side) | (62.5, 0, 37.475) | 5 x 110 x 64.95 |
| wall -y | (-15, -57.5, 37.475) | 160 x 5 x 64.95 |
| wall +y | (-15, 57.5, 37.475) | 160 x 5 x 64.95 |
| handle top bar | (72.5, 0, 62.475) | 15 x 20 x 14.95 |
| handle outer bar | (87.5, 0, 39.975) | 15 x 20 x 59.95 |

The outer-bar box goes down to z = 10 everywhere, so at its inner bottom corner (x 80-82) it is up to
7.5 mm lower than the real rounded bar; the gap above z = 17.5 is exact. Grasp options for the Panda
(max. 80 mm opening): pinch the handle across its 20 mm width (y), or put a finger into the 15 mm gap
from below (whether a Panda finger fits there is not checked).

Mass in Isaac: set the mass and centre of mass explicitly (mass = 292 cm³ x density; centre of mass
(-7.6, 0, 26.6) mm from v3) instead of letting Isaac compute them from the boxes, whose volume is 0.6 %
larger (293.8 cm³, square handle corners).

## Use in the training

The kitchen pour training (`training/hestia_beans_pour`, task `HestiaBeansPour-Franka-Kitchen-Pour`) uses this
container as the bean container. It does not load the STL: `tasks/kitchen_pour/layout.py` rebuilds it from the
measurements above (v2) as 7 boxes (bottom, 4 walls, handle top bar, handle outer bar) and writes them to a small USD
file. The Panda grasps it at the handle from above (fingers across the 20 mm width). Mass in the sim: 0.30 kg, an
estimate for a PLA print; weigh the real one and pass `env.container_mass=<kg>`. Where it stands on the table: see
the training README ("Kitchen beans pour").
