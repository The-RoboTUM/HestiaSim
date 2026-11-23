# AGENTS.md: HestiaSim

Repo overview and rules: README.md. The steps below are the ones to do without being asked.

- Before changing a folder in `components/` or `scenes/`, read its README.md.
- Adding a component or scene: create its README.md from the template in README.md in the same change,
  and add the folder to the layout tree in README.md. Measure size, origin, units and front from the
  files; write "unknown" for anything else (e.g. who made it) and tell the user.
- Changing files in a component or scene (new version, units, position, rename): update that README in
  the same change, and the scene README if a scene uses it.
- File names: `<part>_v<N>.<ext>` (no author, no date); the version counts up per object.
- STL must be in mm. If a received file is in metres or sits at an assembly offset, convert a copy and
  write the conversion under "Changes made here".
- Never copy component geometry into a scene; reference the component's file.
- MuJoCo preview of the U layout (from scenes/u_table_layout):
  `uv run --no-project --with mujoco --with trimesh --with shapely --with mapbox-earcut --with scipy --with networkx --with fast-simplification python preview_U_table_layout.py`
