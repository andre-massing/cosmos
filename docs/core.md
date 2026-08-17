# `cosmos.core`

The `core` subpackage provides the runtime scaffolding shared by every simulation:
mesh/deformation tracking, time stepping, the simulation loop, the field abstraction
used to feed data in and out of PDE models, and small numerical utilities.

## `mesh.py` — `SolverMesh`

Wraps an NGSolve `Mesh` and adds the bookkeeping CoSMoS needs for evolving-domain
problems:

- Caches mesh metadata (`ne`, `nv`, `nedge`, `nface`, `nfacet`, `dim`, curved-element
  `order`) and the sets of volume/boundary/co-dimension-2-boundary region names
  (`vol_markers`, `bnd_markers`, `bbnd_markers`) discovered from the mesh.
- Detects whether the mesh is a pure surface mesh (`is_bnd = True` when `ne == 0`, i.e.
  a manifold embedded with no volume elements) versus a bulk mesh, and builds the
  appropriate `VectorH1` deformation space (`self.V`) over the whole domain
  (`Materials('.*')` or `Boundaries('.*')`).
- Owns the **deformation state**: `curr_deformation` (the deformation being computed for
  the step in progress) and `prev_deformation`, a ring buffer of the last `buffer`
  (see [config.md](config.md)) committed deformations, used by multi-step time
  discretizations that need `X^{n-1}`, `X^{n-2}`, etc.
- `advance_mesh()` temporarily applies `curr_deformation` to the underlying NGSolve mesh
  (`mesh.deformation`) so forms can be assembled on the tentative new configuration;
  `reset_mesh()` restores the last committed deformation. `update_state()` is called once
  a step is accepted: it shifts the `prev_deformation` buffer and commits
  `curr_deformation` as the mesh's deformation.
- `get_state()` / `print_info()` are introspection helpers.

## `time.py` — `SolverTime`

Wraps time-stepping state around an NGSolve `Parameter` so that time-dependent
coefficient functions built from `SolverTime.t`/`SolverTime.t_coef` update automatically
without re-assembling forms from scratch.

- `dt` may be a plain number (fixed time step), or a `Parameter`/`list`/`numpy.ndarray`
  (schedule indexed by `SolverTime.iter`) for variable/adaptive stepping; `dynamic` is set
  accordingly.
- `t_coef` is an optional extra `Parameter` (distinct from `t`) that tracks a
  "look-ahead" time value; `advance_tcoef()`/`reset_tcoef()` move it to `t + dt` and back
  to `t`, mirroring how `SolverMesh.advance_mesh()`/`reset_mesh()` handle the deformation
  — both are used together by PDE models that assemble on a tentative next-step
  configuration inside `Solve()` and roll it back afterwards.
- `postprocess()` advances `iter`, `t`, and refreshes `dt` from the schedule (list/array/
  `Parameter`/constant) supplied at construction; it maintains `prev_dt`/`prev_t` history
  buffers of length `buffer`, analogous to `SolverMesh.prev_deformation`.
- Raises if `final_t < initial_t` or if `dt` is ever set to a non-positive value.

## `solver.py` — `Solver`

The top-level orchestrator of a simulation:

- Constructed from a `SolverMesh` and an optional `SolverTime` (defaults to a
  zero-length, zero-duration `SolverTime` if omitted).
- Maintains `self.pdes`, a name-keyed registry of attached `BasePDEModel` instances, and
  a `SolverIterator` that runs them each step. `_attach_model(pde, order)` is called by
  PDE model constructors to register themselves (raises if the model isn't a
  `BasePDEModel`, `order` isn't an `int`, or the model name is already registered).
- `Solver.__call__()` returns a **generator** (`_generator`) that performs one time step
  per `next()`/loop iteration (see [architecture.md](architecture.md) for the exact
  step sequence) and prints progress if `printing=True`. `Solver.run()` simply exhausts
  that generator to run the simulation to completion. Any exception during the run is
  caught, the last state is flushed to disk, and a `<name>.err` file with the traceback
  is written to `output_folder` before continuing to propagate visibility of the failure
  to the console.
- `output_params(folder, sample_rate)` configures where/how often results are written;
  `save_model_solution(pde, subdivision=0)` opts a specific attached model into VTK
  output (writes `output_fields` of that model) and `_save_pdes_solutions()` performs the
  actual periodic `VTKOutput.Do(...)` calls each step.
- `current_time` is a convenience property returning `self.time.t.Get()`.

## `iterator.py` — `SolverIterator`

Drives the registered PDE models through their lifecycle each time step, in ascending
`model_order`:

1. `preprocess()` — sorts models by `model_order` and calls `PreProcess()` on each.
2. `solve_step()` — calls `Solve()` on each model once. If the iterator was constructed
   with `subiter_bool=True` (i.e. `Solver(..., iter=True)`), it then repeats `Solve()` on
   every model in a fixed-point loop, tracking the relative change in each model's
   solution vector (`Norm(gfu - old_gfu) / len(gfu)`), until all errors drop below
   `1e-8` or 20 sub-iterations are exhausted (raises an exception in the latter case).
   This is how implicitly-coupled multi-physics systems (e.g. a shape-evolution model and
   the `ALEModel` consuming its output) are resolved self-consistently within a step.
3. `postprocess()` — calls `PostProcess()` on each model in the same order.

## `field.py` — `Field`, `InputField`, `OutputField`

A thin abstraction so PDE models can accept a diffusion coefficient, source term, etc.
as a plain number, an NGSolve `CoefficientFunction`/`GridFunction`, *or* a Python
callable returning one of those — and evaluate it uniformly via `field()`.

- **`Field`**: base wrapper; `_eval()`/`__call__()` resolve the stored value regardless
  of which of the three accepted forms it was constructed with.
- **`InputField`**: wraps a model-owned `GridFunction` (`gfu`) that gets refreshed by
  interpolating the current coefficient into it via `update()`
  (`gfu.Set(self(), definedon=domain, dual=True)`). PDE models call
  `BasePDEModel.update_input_fields()` once per solve to keep all of their `InputField`s
  in sync with whatever coefficient (possibly time-dependent) the user last set via
  `set_input_fields()`. The `cf` property setter accepts number/CoefficientFunction/
  GridFunction/callable, matching `Field`'s constructor contract.
- **`OutputField`**: wraps a `GridFunction` that is exposed for output/visualization
  (e.g. registered under `model.output_fields["sol"]` and consumed by
  `Solver.save_model_solution`'s `VTKOutput`). `sample_rate` is reserved for future use.

## `utils.py` — `MandBP`

`MandBP(gfu_vec, dt=None, weights=None, BP=None, MP=False, mass0=None)` post-processes a
raw solution vector (as a NumPy array) to enforce:

- **Bound preservation** (`BP=[lower, upper]`, `MP=False`): simple clipping via
  `np.clip`.
- **Mass-and-bound preservation** (`MP=True`): finds a scalar shift `xsi` (via the
  secant method applied to `F(xsi) = sum(weights * clip(gfu_vec + dt*xsi, *BP)) - mass0`)
  such that clipping the shifted solution to `BP` reproduces the target total mass
  `mass0` under the given quadrature `weights` (typically a mass-lumped mass matrix's
  diagonal). This is the mechanism used by `mass_preserving=True` PDE model options (see
  [pde_models.md](pde_models.md)) to keep a phase/concentration field within physical
  bounds while conserving its total integral, following Zhang & Shu-style bound- and
  mass-preserving limiters.

Used internally by `Solve()` in the ADR and Cahn-Hilliard model families whenever
`input_params["bounds"]` and/or `input_params["mass_preserving"]` are set.
