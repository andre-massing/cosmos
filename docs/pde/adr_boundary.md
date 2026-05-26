# ADR boundary models

**Sources:**
- `src/cosmos/pde/adr/boundary/adr_boundary_system_bdf1_model_nostab.py`
- `src/cosmos/pde/adr/boundary/adr_boundary_system_bdf1_model_stab.py`

Both files expose a class named `ADRBoundarySystemBDF1Model`; import the one
that matches the desired stabilisation strategy.

---

## `ADRBoundarySystemBDF1Model` — no stabilisation

```python
# import path
from cosmos.pde.adr.boundary.adr_boundary_system_bdf1_model_nostab import (
    ADRBoundarySystemBDF1Model,
)
```

BDF1 advection–diffusion–reaction system on a surface compartment (no
stabilisation).

Discretises a coupled system of `dim` scalar species on a boundary domain using
H1 finite elements and a first-order backward-difference (BDF1) time scheme.
Supports Dirichlet and Neumann boundary conditions, user-defined nonlinear
reaction terms via `add_nonlinearity()`, mass preservation, and
bound-preserving post-processing.

The finite-element space is `(H1)^dim`; one H1 scalar space per species. The
bilinear form uses a symmetric interior-penalty Galerkin (SIPG) method for
diffusion and upwind terms for advection, both living on the surface mesh (via
`.Trace()` operators).

---

## `ADRBoundarySystemBDF1Model` — gradient-jump stabilisation

```python
from cosmos.pde.adr.boundary.adr_boundary_system_bdf1_model_stab import (
    ADRBoundarySystemBDF1Model,
)
```

BDF1 advection–diffusion–reaction system on a surface compartment with
gradient-jump stabilisation.

Augments the standard SIPG bilinear form with gradient-jump penalty terms
scaled by the local Péclet number, improving robustness in advection-dominated
regimes. Uses a mixed H1 × NormalFacetSurface finite element space; the
stabilisation flux unknowns are eliminated locally at assembly time. All other
features match the unstabilised variant.

---

## Constructor (both variants)

```python
ADRBoundarySystemBDF1Model(
    name: str = 'ADRBoundarySystemBDF1Model',
    model: CosmosModel = None,
    compartment: CosmosCompartment = None,
    **kwargs,
)
```

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Identifier shown in VTK output names. |
| `model` | `CosmosModel` | Owning model. |
| `compartment` | `CosmosCompartment` | Surface compartment. Must be a boundary compartment (`is_bnd = True`). |
| `dim` | `int` *(required via kwargs)* | Number of coupled scalar species. |

**Raises:** `Exception` if `dim` is absent or not a number.

---

## Parameters

Access via `pde.set_params(key=value)` after construction.

### Discretisation

| Key | Type | Default | Description |
|---|---|---|---|
| `fes_order` | `int` | `1` | Polynomial order of the H1 space. |
| `Dir_bnd` | `str` | `""` | Pipe-separated co-boundary names for Dirichlet conditions. |
| `Neu_bnd` | `str` | `""` | Pipe-separated co-boundary names for Neumann conditions. |

### Per-species (replace `{i}` with `1`, `2`, …, `dim`)

| Key | Type | Default | Description |
|---|---|---|---|
| `u0_{i}` | `CoefficientFunction` | `CF(0)` | Initial condition for species *i*. |
| `b_{i}` | `Field` | `CF((0,…))` | Advection velocity for species *i* (vector). |
| `d_{i}` | `Field` | `CF(0)` | Diffusion coefficient for species *i*. |
| `c_{i}` | `Field` | `CF(0)` | Reaction coefficient (multiplies the mass-matrix term) for species *i*. |
| `rhs_{i}` | `Field` | `CF(0)` | Source/right-hand-side for species *i*. |
| `u_bnd_{i}` | `Field` | `CF(0)` | Dirichlet boundary value for species *i*. |
| `gradu_bnd_{i}` | `Field` | `CF((0,…))` | Neumann boundary flux gradient for species *i*. |
| `mass_preserving_{i}` | `bool` | `False` | Enforce total-mass conservation via a secant post-correction. |
| `bounds_{i}` | `[lo, hi] \| None` | `None` | Enforce pointwise bounds `[lo, hi]` via clipping post-correction. |

---

## Methods

### `Initialize`

Assembles the bilinear form (including BDF1 mass-matrix term) and inverts it.
Sets initial conditions for all species.

---

### `PreProcess`

Snapshots `gfu_old ← gfu`.

---

### `Solve`

Updates all parameter GridFunctions, reassembles the system, solves, and
optionally applies bound/mass post-corrections.

---

### `PostProcess`

For 2D meshes: projects each species component into a full H1 GridFunction
suitable for VTK output.

---

### `add_nonlinearity`

```python
def add_nonlinearity(
    target: int,
    expression: str,
    map: dict = {},
) -> None
```

Append a nonlinear reaction term to species `target`.

| Parameter | Type | Description |
|---|---|---|
| `target` | `int` | Index (1-based) of the species to which the term is added. |
| `expression` | `str` | Python expression string evaluated with `eval`. The current solution components are available as `u1`, `u2`, …; the test functions as `v1`, `v2`, …. Standard NGSolve symbols (`sin`, `cos`, `exp`, `IfPos`, `x`, `y`, `z`) are in scope. |
| `map` | `dict` | Additional name → NGSolve object mappings injected into the `eval` environment. |

**Example:**

```python
adr.add_nonlinearity(
    target=1,
    expression="u1 * (1 - u1) * v1",
)
```
