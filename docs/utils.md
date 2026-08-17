# `cosmos.utils`

Support tooling used to build test/demo geometries and to derive manufactured solutions
for convergence testing; not part of the runtime simulation loop itself.

## Mesh generation — `generate_meshes.py`

A library of functions that build standard test geometries as NGSolve `Mesh` objects,
used throughout `tests/` and `reproduction/`. Every geometry function follows the naming
convention `generate_boundary_*` (produces a pure surface mesh, `mesh.ne == 0`, suitable
for boundary/surface PDE models) or `generate_volume_*` (produces a bulk mesh with
volume elements). They share a common `Meshing(geo, maxh, order_g, vol_or_bnd)` helper
that calls `geo.GenerateMesh(...)` (stopping at `MeshingStep.MESHSURFACE` for boundary
meshes) and curves the mesh to `order_g`.

Available geometries include: `generate_boundary_arc`, `generate_boundary_plane`,
`generate_boundary_circle` (circle embedded in 3D), `generate_volume_circle` (2D disk),
`generate_boundary_sphere`, `generate_boundary_ellipse`, `generate_boundary_sigar`
(capsule/stadium shape), `generate_boundary_half_sphere`, `generate_volume_ball`,
`generate_boundary_cylinder`, `generate_boundary_torus`/`generate_boundary_half_torus`,
`generate_boundary_box`/`generate_boundary_smoothed_box` (filleted-edge box), and
`import_stl_mesh` (mesh an externally supplied `.stl` file, e.g. the Stanford bunny used
in `reproduction/scripts/4_5_Stanford_Bunny`).

Most accept `maxh` (target mesh size) and `order_g` (curved-element order used for
`mesh.Curve(order_g)`); several also let the caller name the (B)Boundary region created
(`bnd_name`/`bbnd_name`) so PDE models can target boundary conditions at it via
`domain=...`/`Dir_bnd=...`/`clamped_bnd=...` parameters.

## Manufactured solutions — `manufactured_solution_tools.py`

A SymPy-based helper module (written in Jupytext `# %%` cell format) for deriving
right-hand sides of manufactured-solution convergence tests. Given an analytic solution
`u(x, y, z)` and an implicit surface description `phi(x, y, z) = 0`, it computes the
surface normal and the corresponding forcing term needed to make `u` an exact solution of
a surface PDE.

Differential-operator helpers (built directly on SymPy, since SymPy has no native support
for these): `Grad` (gradient of a scalar or a vector field, returned as a `Matrix`),
`Div` (divergence of a vector/tensor field), `Laplace` (full/ambient Laplacian),
`Hess` (Hessian matrix), `Tr` (matrix trace), `Project_Grad` (tangential/surface gradient
`grad(f) - (n . grad(f)) n`), `compute_normal` (unit normal `grad(phi)/|grad(phi)|`), and
`LaplaceBeltrami` (the surface Laplacian, computed from the ambient Laplacian, Hessian,
and normal via the identity documented in the module's markdown cell).

- **`get_solution_str(u_str, levelset_str, epsilon_str="1", b_str=["0","0","0"],
  c_str="1")`**: given string expressions for `u`, the level set `phi`, and the PDE
  coefficients `epsilon` (diffusivity), `b` (advection), `c` (reaction), returns the
  matching right-hand side `f` and tangential gradient of `u` (as strings, ready to be
  fed into `CoefficientFunction`/parsed by NGSolve) for the surface PDE
  `-epsilon*Delta_Gamma(u) + c*u + b . grad_Gamma(u) = f`.
- **`eprint`/`vec_simplify`**: pretty-printing helpers for interactively inspecting the
  derived symbolic expressions in a notebook (LaTeX via `IPython.display` for small
  expressions, SymPy's unicode pretty-printer for large ones).

This module is intended to be run interactively (e.g. via Jupytext/Jupyter) to derive the
`rhs`/boundary-data expressions later hard-coded into convergence tests such as those
under `tests/pde/convergence/`.

## Math helpers — `tools.py`

- **`gradient(f, P)`**: a general tangential-gradient operator for scalar or vector
  NGSolve `CoefficientFunction`s `f`, built from raw symbolic differentiation
  (`f.Diff(x)`, etc.) and projected by the tangential projector `P`. Handles 2D and 3D
  ambient space; for vector-valued `f` it differentiates component-wise and reassembles
  the Jacobian. Used where NGSolve's built-in `grad`/`Trace()` machinery isn't directly
  applicable (e.g. differentiating a symbolic expression rather than a `GridFunction`).
- **`compute_stab_mc(data, gfu, params)`**: solves an auxiliary mixed mean-curvature
  problem (mirroring the mixed formulation in
  `mean_curvature_boundary_stab_bdf1_model.py`) to compute a stabilized *initial* mean
  curvature field, including an optional facet-jump stabilization term
  (`params['stab']`) and an optional clamped-boundary conormal condition
  (`params['clamped_bnd']`), and writes the result into the caller-supplied `gfu`. Used
  to bootstrap the curvature-dependent Willmore/mean-curvature models' `gfu_k_old` state
  outside of the standard `A_mc`/`F_mc` projection built into those models' own
  `__init__`.

A currently commented-out `params_check` helper (validates a `params` dict against an
accepted-keys/defaults list) is left in place as dead code in the source and is not part
of the public API.
