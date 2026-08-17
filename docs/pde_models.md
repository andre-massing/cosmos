# `cosmos.pde`

Every concrete PDE solver in CoSMoS lives under `cosmos.pde`, grouped into one
subpackage per physics. All of them derive from `BasePDEModel` (`cosmos/pde/base.py`)
and follow the same construction/lifecycle pattern described in
[architecture.md](architecture.md). This document describes the base contract, then each
physics family and the differences between its BDF1/BDF2/stabilized/preserving variants.

## `base.py` — `BasePDEModel`

Abstract base class (`abc.ABC`) all models implement:

- **State set up in `__init__`** (by subclasses, since `BasePDEModel.__init__` only
  initializes shared bookkeeping): `name`, `input_params` (a plain `dict` of scalar
  options), `input_fields`/`output_fields` (`dict[str, InputField/OutputField]`, see
  [core.md](core.md#fieldpy--field-inputfield-outputfield)), `cfg` (the shared
  `Config`, see [config.md](config.md)), and VTK bookkeeping (`save`, `vtk`, `VorB`).
- **Abstract methods** every subclass must implement: `PreProcess()`, `Solve()`,
  `PostProcess()` — called once per time step, in that order, by `SolverIterator`.
- **`set_input_fields(dict)`** / **`set_input_params(dict)`**: validate that each key
  already exists in `input_fields`/`input_params` (raising otherwise) and update the
  corresponding entry — `set_input_fields` sets `InputField.cf` (which callers then push
  into the underlying `GridFunction` via `update_input_fields()`), `set_input_params`
  overwrites the raw value directly.
- **`update_input_fields()`**: calls `.update()` on every registered `InputField`,
  refreshing its backing `GridFunction` from the current coefficient. Nearly every
  concrete model's `Solve()` calls this first.
- **`get_output_fields()`**: returns `self.output_fields`, consumed by
  `Solver.save_model_solution`.

## Shared implementation pattern

Every concrete model's `__init__` follows the same recipe:

1. Validate `domain` against `solver.mesh.vol_markers`/`bnd_markers` and resolve it to an
   NGSolve region (`solver.ngsmesh.Materials(...)` or `.Boundaries(...)`).
2. Call `solver._attach_model(self, model_order)` to register with the `Solver`.
3. Populate `self.input_params` with named defaults, then apply any user overrides via
   `self.set_input_params(input_params)`.
4. Build one or more finite element spaces (`H1`, `VectorH1`, product/compound spaces via
   `*`/`CompressCompound`, optionally `Periodic`/`Compress`-wrapped), a `GridFunction`
   solution `gfu` (plus `gfu_old` holding the previous time level), and register
   `output_fields`.
5. Create `InputField`s for every physical coefficient/right-hand-side/boundary datum the
   model needs, each backed by its own `GridFunction`.
6. Assemble a `BilinearForm` `self.A` and `LinearForm` self.F` over the (deformed) domain,
   invert `A` once (`self.A.mat.Inverse(freedofs=...)`) since the system is linear in the
   unknown at each step — a linear solve is then just a mat-vec product with the cached
   inverse, refreshed by `.Update()` whenever `A` is re-assembled (its coefficients are
   time/state-dependent, e.g. through `gfu_old` or the deformation).
7. `PreProcess()` typically snapshots `gfu_old.vec.data = gfu.vec.data` (and, where
   relevant, checks/initializes bound- or mass-preservation state on the first
   iteration).
8. `Solve()` refreshes input fields, re-assembles `A`/`F` (their coefficients depend on
   the just-updated inputs, the current deformation, and/or `gfu_old`), solves
   `gfu.vec.data = invA * F.vec`, and — for models exposing `bounds`/`mass_preserving`
   options — post-processes the raw solution through `MandBP` (see
   [core.md](core.md#utilspy--mandbp)).
9. `PostProcess()` is a no-op in most models (state advancement for the *mesh* happens in
   `ALEModel.PostProcess`, not in the physics models themselves).
10. Convenience `@property`/`@property.setter` pairs expose named solution components
    (e.g. `.sol`, `.phase`, `.potential`, `.displacement`, `.mean_curvature`).

## Advection-Diffusion-Reaction — `cosmos.pde.adr`

Solves, in each variant, a linear parabolic advection-diffusion-reaction equation

```
du/dt + div(b u) + c u - div(d grad u) = rhs
```

on a (deforming) volume or boundary domain, discretized with SIP-DG-style interior
penalty terms on Dirichlet boundaries and upwinding of the advective flux.

- **`volume/adr_volume_bdf1_model.py` — `ADRVolumeBDF1Model`**: bulk-domain, first-order
  BDF (implicit Euler) in time. Options: `Neu_bnd`/`Dir_bnd` region names, `periodic`,
  `mass_preserving`, `bounds` (a `[low, high]` pair enforced via `MandBP`), `fes_order`,
  `u0` (initial condition). Input fields: `b` (advective velocity), `d` (diffusivity),
  `c` (reaction coefficient), `rhs`, `gradu_bnd` (Neumann flux), `u_bnd` (Dirichlet
  value).
- **`volume/adr_volume_bdf2_model.py` — `ADRVolumeBDF2Model`**: same equation, second-order
  BDF2 time discretization (needs two previous time levels, hence `Config.buffer >= 2`).
- **`volume/adr_volume_stab_bdf1_model.py` / `..._stab_bdf2_model.py`**: as above plus a
  jump-penalization stabilization term (`stab * jump(grad u) * jump(grad v)` over facets,
  using `dgjumps=True` spaces) proportional to `Norm(b) * h**2`, improving robustness in
  advection-dominated regimes.
- **`boundary/adr_boundary_bdf1_model.py` / `_bdf2_model.py` / `_stab_bdf1_model.py` /
  `_stab_bdf2_model.py`**: the boundary-domain (surface PDE / codimension-1) counterparts
  of the four volume variants above, using tangential (`.Trace()`) gradients and
  facet-based (`FacetSurface`/`NormalFacetSurface` or, in 2D ambient space, nodal)
  representations of the co-dimension-2 boundary conditions instead of volumetric
  Dirichlet/Neumann terms.

All eight variants share the `mass_preserving`/`bounds` post-processing machinery
described above.

## Cahn-Hilliard phase separation — `cosmos.pde.cahn_hilliard`

Solves the (surface) Cahn-Hilliard system for a phase field `u` and chemical potential
`w`:

```
du/dt + div(b u) - div(M grad w) = rhs_u
w + sigma*epsilon*Delta u ... = rhs_w     (double-well potential term, linearized in time)
```

split into a mixed two-field formulation (`fes = _fes * _fes`) so only first derivatives
appear. The double-well potential `W` is linearized about `gfu_u_old` (`dW`, `ddW`
closures inside each `__init__`) — i.e. each step is a linearly-implicit (semi-implicit,
convex-splitting-style) BDF1 step, not a fully nonlinear solve.

- **`volume/cahn_hilliard_volume_bdf1_model.py` — `CahnHilliardVolumeBDF1Model`**:
  bulk-domain, uses a logarithmic Flory-Huggins-style potential
  (`dW = theta1*(log(1+u) - log(1-u)) - theta2*u`, encoded directly in the assembled
  forms). Options include `epsilon`, `theta1`, `theta2`, `mass_preserving`, `bounds`.
- **`volume/cahn_hilliard_volume_aland_bdf1_model.py` —
  `CahnHilliardVolumeAlandBDF1Model`**: bulk-domain, uses the smooth polynomial
  double-well `W(u) = (u^2-1)^2/4` (Aland-style splitting scheme). Options include `M`
  (mobility), `epsilon`, `sigma` (surface tension), `Neu_bnd_phase`/`Neu_bnd_potential`.
- **`boundary/cahn_hilliard_boundary_bachini_bdf1_model.py` —
  `CahnHilliardBoundaryBachiniBDF1Model`**: the surface-PDE analogue of the Aland scheme
  (same polynomial double well, tangential operators), following Bachini et al.'s
  surface Cahn-Hilliard discretization; exposes an `energy` property integrating the
  Ginzburg-Landau free energy over the boundary.
- **`boundary/cahn_hilliard_boundary_bachini_log_bdf1_model.py` —
  `CahnHilliardBoundaryBachiniLogBDF1Model`**: the same surface scheme but with the
  logarithmic Flory-Huggins potential in place of the polynomial one (mirroring the
  volume BDF1 vs. Aland-BDF1 split).

All variants support `bounds`/`mass_preserving` post-processing via `MandBP`, restricted
to `fes_order == 1`.

## Mean curvature flow — `cosmos.pde.mean_curvature`

Solves surface mean curvature flow via a mixed formulation that solves simultaneously for
the position update (displacement `D`) and the (vector) mean curvature `kappa`:

```
dD/dt = kappa            (up to the coefficient "kappa" scaling below)
kappa = Delta_Gamma X     (mean curvature = tangential Laplacian of the identity)
```

- **`mean_curvature_boundary_bdf1_model.py` — `MeanCurvatureBoundaryBDF1Model`**: BDF1 in
  time, mass-lumped (`ds_lumped`, using low-order quadrature) for the reaction-type
  coupling terms between `D` and `kappa`. Option: `kappa` (a scalar flow-speed
  coefficient).
- **`mean_curvature_boundary_stab_bdf1_model.py` —
  `MeanCurvatureBoundaryStabBDF1Model`**: adds a third unknown — the facet-normal
  derivative jump of the curvature (`NormalFacetSurface` space) — and a
  jump-penalization stabilization term (coefficient `stabilization`) analogous to the
  ADR "stab" variants, to control spurious oscillations of the discrete curvature; only
  implemented for 3D ambient space (2D/1D-manifold case raises `Exception`).

Both expose `displacement`/`mean_curvature` properties (note: the corresponding
`@x.setter` methods are mis-named `phase`/`potential` rather than `displacement`/
`mean_curvature` in the source — a pre-existing quirk carried over unchanged from
Cahn-Hilliard-family code, not something to rely on).

## Willmore / Helfrich bending flow — `cosmos.pde.willmore`

Solves L²-gradient flow of the Helfrich bending energy
`E = 0.5 * kappa_elastic * integral((H - H0)^2)` (mean curvature `H`, spontaneous
curvature `H0`, elasticity modulus `kappa_elastic`), via a mixed formulation splitting the
fourth-order-in-space problem into a system for the displacement `D` and an auxiliary
variable `Y = kappa_elastic * (H - H0 n)`. All variants first solve an auxiliary "initial
mean curvature" projection problem (`A_mc`/`F_mc`, mirroring
`MeanCurvatureBoundaryBDF1Model`'s mixed form) to (re-)initialize `gfu_k_old` — either
once, at `iter == 0`, or every step if `input_params["autoupdate"] = True`.

- **`willmore_boundary_bdf1_model.py` — `WillmoreBoundaryBDF1Model`**: the baseline BDF1
  scheme. Options: `clamped_bnd` (a BBoundary name where the surface's co-normal is
  pinned to `clamped_conormal`, for open/clamped membranes), `clamped_conormal`,
  `autoupdate`. Exposes an `energy` property.
- **`willmore_boundary_ap_bdf1_model.py` — `WillmoreBoundaryAPBDF1Model`** (**A**rea
  **P**reserving): adds a scalar Lagrange multiplier `lambda_h` (a `NumberSpace`
  unknown) enforcing conservation of surface area, solved via a fixed-point loop inside
  `Solve()` (re-evaluates `lambda_h` from a Rayleigh-quotient-like integral, re-solves the
  linear system, and repeats until `lambda_h` converges or 10 iterations are exhausted,
  raising if it doesn't converge). Not compatible with `clamped_bnd`.
- **`willmore_boundary_vp_bdf1_model.py` — `WillmoreBoundaryVPBDF1Model`** (**V**olume
  **P**reserving): the analogous construction, but the Lagrange multiplier instead
  enforces conservation of enclosed volume.
- **`willmore_boundary_apvp_bdf1_model.py` — `WillmoreBoundaryAPVPBDF1Model`** (**A**rea-
  **and-V**olume-**P**reserving): uses *two* Lagrange multipliers (area and volume) and a
  2-D fixed-point/Newton-style update to satisfy both constraints simultaneously each
  step.
- **`willmore_boundary_inex_bdf1_model.py` — `WillmoreBoundaryInexBDF1Model`**
  (**Inex**act): a variant of the base BDF1 scheme using an inexact/linearized treatment
  of one of the nonlinear coupling terms, trading some accuracy for a cheaper (no
  fixed-point sub-loop) assembly compared to the AP/VP/APVP variants.

All Willmore variants expose `displacement`, `mean_curvature`, and `energy` properties,
and are driven, in practice, together with an `ALEModel` that consumes `.displacement` to
move the mesh each step (see [coupling.md](coupling.md) and
`tests/pde/physical/willmore/test_helfrich_spine.py` for a worked example iterating over
several Willmore variants).

## Choosing a variant

- Prefer `BDF2` over `BDF1` for smoother/more accurate long-time integration once a
  simulation is past its first two steps (`Config.buffer` must be `>= 2`).
- Add the `Stab` variant (ADR, mean curvature) when advection dominates diffusion or the
  discrete curvature exhibits mesh-induced oscillations.
- Use `mass_preserving=True`/`bounds=[lo, hi]` (ADR, Cahn-Hilliard) whenever the modeled
  quantity has a physical conservation law and/or must stay within a known range (e.g. a
  concentration in `[0, 1]`).
- For Willmore/Helfrich flow, pick `AP`/`VP`/`APVP` when the simulated membrane must
  conserve area and/or enclosed volume as it evolves (e.g. a closed vesicle), and the
  plain `BDF1` or `Inex` variant otherwise.
