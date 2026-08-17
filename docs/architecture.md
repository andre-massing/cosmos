# Architecture

CoSMoS wires together a small set of core abstractions that any simulation script
composes: a **mesh**, a **time manager**, a **solver**, and one or more **PDE models**
attached to that solver. This document explains the role of each piece and the order
in which they interact during a simulation.

## Core abstractions

- **`SolverMesh`** (`cosmos.core.mesh`) wraps an NGSolve `Mesh` and tracks the mesh
  *deformation* — the displacement field that moves the mesh nodes over time. It keeps a
  short history buffer of past deformations (`prev_deformation`) so that multi-step time
  discretizations (e.g. BDF2) can access previous states. It works uniformly for pure
  surface meshes (`ne == 0`, i.e. a 2D manifold in 3D space) and bulk/volume meshes.

- **`SolverTime`** (`cosmos.core.time`) is a thin wrapper around an NGSolve `Parameter`
  holding the current simulation time `t` and time-step `dt`. `dt` may be constant, or
  advance according to a list/array/`Parameter` for adaptive stepping. It also keeps a
  history buffer of past `dt`/`t` values, mirroring `SolverMesh`.

- **`BasePDEModel`** (`cosmos.pde.base`) is the abstract contract every physics model
  implements: `PreProcess()`, `Solve()`, `PostProcess()`. A model owns its own finite
  element space(s), a `GridFunction` solution (`gfu`), and dictionaries of named
  `InputField`/`OutputField` objects (`cosmos.core.field`) used to feed in
  space/time-varying coefficients and to expose solution fields for visualization/output.

- **`Solver`** (`cosmos.core.solver`) owns a `SolverMesh`, a `SolverTime`, and a registry
  of attached `BasePDEModel` instances (`self.pdes`). Models register themselves via
  `Solver._attach_model()`, typically from their own `__init__`, passing a `model_order`
  that determines the order in which they run within a time step. `Solver` is a Python
  generator: calling it (`for _ in solver(): ...`) advances the simulation one time step
  per iteration, or `solver.run()` can be used to run to completion without inspecting
  intermediate steps.

- **`SolverIterator`** (`cosmos.core.iterator`) is used internally by `Solver` to run the
  registered PDE models in `model_order` each time step: `preprocess()` →
  `solve_step()` → `postprocess()`. If constructed with `iter=True`, `solve_step()`
  repeats `Solve()` on every model (a fixed-point/Picard sub-iteration) until the change
  in each model's solution vector falls below a tolerance or a maximum of 20
  sub-iterations is reached — this is how coupled multi-physics systems (e.g. a curvature
  flow model driving an ALE mesh model) are resolved implicitly within one time step.

- **`ALEModel`** (`cosmos.coupling.ale_model`) is itself a `BasePDEModel` that computes
  the mesh deformation increment for the *next* step from displacement fields prescribed
  by other models (e.g. the displacement produced by a Willmore flow model), optionally
  redistributing mesh nodes tangentially to avoid mesh degeneration as the surface
  evolves. See [coupling.md](coupling.md).

## Execution flow of a simulation

1. **Build a mesh.** Load or generate an NGSolve `Mesh` (see
   [utils.md](utils.md#mesh-generation)) and wrap it in a `SolverMesh`.
2. **Configure time stepping.** Create a `SolverTime` with the desired `dt`,
   `initial_t`, `final_t`.
3. **Create the `Solver`.** Pass it the `SolverMesh` and `SolverTime`.
4. **Attach PDE models.** Instantiate one or more `BasePDEModel` subclasses
   (e.g. `WillmoreBoundaryBDF1Model`, `ADRVolumeBDF1Model`, `ALEModel`, ...), each
   passing the `Solver` and a `model_order`. Each model's constructor assembles its
   finite element spaces and bilinear/linear forms and calls
   `solver._attach_model(self, model_order)` to register itself.
5. **Configure input fields/parameters** on each model via `set_input_fields` /
   `set_input_params` (e.g. diffusion coefficients, right-hand sides, boundary data).
6. **(Optional) configure output.** `solver.output_params(folder, sample_rate)` plus
   `solver.save_model_solution(model)` per model registers periodic VTK output.
7. **Run the simulation.** Either `solver.run()` to run to completion, or iterate the
   generator `for _ in solver(): ...` to inspect/act on intermediate state (as the test
   suite in `tests/pde/physical` does to log integrated quantities like area, volume, or
   energy at each step).

Each call to the `Solver` generator performs, in order:

```
SolverTime.preprocess()               # currently a no-op hook
SolverIterator.preprocess()             # PDE models are ordered by model_order,
                                         # then each model's PreProcess() is called
                                         # (typically: snapshot old solution -> gfu_old)
SolverIterator.solve_step()             # each model's Solve() is called in order;
                                         # optionally repeated (sub-iteration) until
                                         # convergence across all models
SolverIterator.postprocess()            # each model's PostProcess() is called
                                         # (e.g. ALEModel advances the mesh state)
SolverTime.postprocess()                # advance t by dt, refresh dt from schedule
Solver._save_pdes_solutions()           # write VTK output for models with save == True
```

## Why `model_order` matters

PDE models are solved in ascending `model_order` within a single time step. This lets a
simulation script sequence dependent physics correctly — for example, solving a curvature
flow model (order 1) before the `ALEModel` (order 2) that consumes its displacement field
to move the mesh, and finally any transport model (order 3) that needs the updated mesh
deformation. When `Solver(iter=True)` is used, this ordering is repeated every
sub-iteration until the coupled system's solutions stabilize.

## Deformed-domain integration

Nearly every PDE model integrates over a *deformed* configuration by passing
`deformation=...` to NGSolve's `dx`/`ds` integrators, using either the current
deformation (`solver.mesh.curr_deformation`, updated in-place during `Solve()` via
`solver.mesh.advance_mesh()`/`reset_mesh()`) or the last committed deformation
(`solver.mesh.prev_deformation[-1]`). This is how the finite element forms are evaluated
on the moving membrane/domain without remeshing at every step.

## Naming conventions in `cosmos.pde`

Model class and file names encode three things:

- **Physics**: `ADR` (advection-diffusion-reaction), `CahnHilliard`, `MeanCurvature`,
  `Willmore`.
- **Where it is posed**: `Volume` (bulk/material domain, `VOL`) or `Boundary`
  (codimension-1 surface, `BND`).
- **Time discretization / variant**: `BDF1`/`BDF2` (first/second-order backward
  differentiation formula), `Stab` (adds a stabilization/penalty term, e.g. jump
  penalization or curvature-gradient stabilization), and, for Willmore flow specifically,
  `AP`/`VP`/`APVP` (area-/volume-/area-and-volume-preserving variants via a Lagrange
  multiplier) and `Inex` (an inexact/linearized variant). See
  [pde_models.md](pde_models.md) for details of each family.

All concrete models derive from `BasePDEModel` and are re-exported from the top-level
`cosmos` package (see `src/cosmos/__init__.py`), so user code simply does
`from cosmos import WillmoreBoundaryBDF1Model, Solver, SolverMesh, SolverTime`.
