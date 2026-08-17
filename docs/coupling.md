# `cosmos.coupling`

## `ale_model.py` — `ALEField`, `ALEModel`

Implements the Arbitrary Lagrangian-Eulerian (ALE) mesh-motion side of a moving-domain
simulation: given a displacement prescribed on part of the domain (typically produced by
a shape-evolution model such as `WillmoreBoundaryBDF1Model`), it extends that
displacement into the mesh interior/whole surface as a harmonic extension, so the mesh
stays valid (non-degenerate) as the domain deforms.

### `ALEField`

A `Field` (see [core.md](core.md#fieldpy--field-inputfield-outputfield)) representing one
prescribed-displacement contribution, either on a boundary region (`VorB=BND`) or a
material/volume region (`VorB=VOL`). Beyond wrapping the coefficient, it optionally
implements **tangential mesh redistribution** — reparametrizing the surface mesh nodes
tangentially (independent of the normal motion that changes the shape) to keep element
quality high over long simulations, via `redistribute_type`:

- **`"DuanLi"`**: solves a mixed harmonic-map-style problem for a tangential
  reparametrization "wind" field `w` on the *fixed* reference configuration
  (`deformation0`, i.e. no deformation applied), balancing it against the current
  material deformation `mat_def`.
- **`"MDR"`** (Mesh Distribution / time-weighted variant): the same style of problem but
  assembled on the last committed deformation (`solver.mesh.prev_deformation[-1]`) and
  scaled by `1/dt`, making the redistribution rate track the time-step size.

`ALEField.update()` refreshes the prescribed displacement `GridFunction`
(`ale_displ`) from the current coefficient and, if redistribution is enabled, solves the
auxiliary mixed problem (`A_pp`/`F_pp`) each call and adds the resulting tangential
"wind" correction into `ale_displ`.

### `ALEModel`

A `BasePDEModel` that aggregates all prescribed `ALEField`s registered on it and computes
the actual mesh deformation increment to apply:

- **`set_bnd_displacement(coef, domain, clamped_bnd='', redistribute=False,
  redistribute_type='DuanLi')`**: registers a boundary-region displacement field. Raises
  if a volume displacement was already prescribed on this model (the two modes are
  mutually exclusive per model instance), or if `domain`/`clamped_bnd` aren't valid
  region names for the mesh.
- **`set_vol_displacement(coef, domain)`**: the volume-domain counterpart (only valid on
  bulk meshes).
- **`Solve()`**: sums the (possibly redistributed) displacement contributions from every
  registered field into `self.gfu`, and — for the boundary case on a bulk mesh (`ne !=
  0`) — projects that boundary displacement into a discrete harmonic extension over the
  whole volume by solving a Laplace problem (`self.A`/`self.invA`, assembled once in
  `__init__` as `InnerProduct(grad(u), grad(v))*dx` with the boundary as a Dirichlet
  condition) so the interior mesh moves smoothly along with the prescribed boundary
  motion. The tangential "wind" correction (`self.tot_wind`) is extended the same way.
  Finally, `solver.mesh.curr_deformation` is set to
  `prev_deformation[-1] + gfu` — i.e. the *tentative* new deformation for this step,
  which other models can read via `solver.mesh.advance_mesh()` before it is committed.
- **`PostProcess()`**: calls `solver.mesh.update_state()` to commit `curr_deformation` as
  the mesh's new, accepted deformation and shift the `prev_deformation` history buffer —
  this is the point in the simulation loop where the mesh actually moves.
- **Derived kinematic properties**: `displacement` (`gfu`, the raw increment just
  solved for), `ale_vel` (`gfu / dt`, the ALE grid velocity), `wind` (`tot_wind / dt`,
  the tangential redistribution velocity), `mat_vel` (`(tot_wind + gfu) / dt`, the
  material point velocity including redistribution).

### Typical usage

An `ALEModel` is attached to the `Solver` at a `model_order` *after* the shape-evolution
model whose displacement it consumes, so that by the time `ALEModel.Solve()` runs within
a step, the upstream model's `.displacement` field already reflects that step's update:

```python
willmore = WillmoreBoundaryBDF1Model(solver, model_order=1, input_params={"autoupdate": True})
ale = ALEModel(solver, model_order=2)
ale.set_bnd_displacement(willmore.displacement, "default", redistribute=True, redistribute_type="DuanLi")
```

See `tests/pde/physical/willmore/test_helfrich_spine.py` for a complete worked example
that exercises every Willmore variant together with `ALEModel` and both redistribution
strategies.
