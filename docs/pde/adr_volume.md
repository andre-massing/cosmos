# ADRVolumeSystemBDF1Model

**Sources:**
- `src/cosmos/pde/adr/volume/adr_volume_system_bdf1_model.py` *(primary)*
- `src/cosmos/pde/adr/volume/adr_volume_system_bdf1_model_v1.py` *(v1 — full reassembly each step)*

```python
from cosmos.pde.adr.volume.adr_volume_system_bdf1_model import ADRVolumeSystemBDF1Model
# or
from cosmos.pde.adr.volume.adr_volume_system_bdf1_model_v1 import ADRVolumeSystemBDF1Model
```

BDF1 advection–diffusion–reaction system on a volume compartment.

Discretises a coupled system of `dim` scalar species on a volumetric domain
using H1 finite elements with interior-penalty (SIP) stabilisation and a
first-order backward-difference time scheme. Supports Dirichlet, Neumann, and
total-flux boundary conditions, nonlinear reaction terms, mass preservation,
and bound-preserving post-processing.

### Primary vs v1

| Variant | Assembly strategy |
|---|---|
| Primary (`_model.py`) | Assembles the stiffness matrix once in `Initialize()` and only re-assembles the parameter-dependent part each `Solve()`. |
| v1 (`_model_v1.py`) | Reassembles the full system matrix at every `Solve()` call. Simpler code; recommended when coefficients change every step in a way that affects the stiffness matrix structure. |

---

## Constructor

```python
ADRVolumeSystemBDF1Model(
    name: str = 'ADRVolumeSystemBDF1Model',
    model: CosmosModel = None,
    compartment: CosmosCompartment = None,
    **kwargs,
)
```

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Identifier shown in VTK output names. |
| `model` | `CosmosModel` | Owning model. |
| `compartment` | `CosmosCompartment` | Volume compartment (`is_vol = True`). |
| `dim` | `int` *(required via kwargs)* | Number of coupled scalar species. |

**Raises:** `Exception` if `dim` is absent or not a number.

---

## Parameters

Access via `pde.set_params(key=value)` after construction.

### Discretisation

| Key | Type | Default | Description |
|---|---|---|---|
| `fes_order` | `int` | `1` | Polynomial order of the H1 space. |
| `Dir_bnd` | `str` | `""` | Named boundary for Dirichlet conditions. |
| `Neu_bnd` | `str` | `""` | Named boundary for Neumann conditions. |

### Per-species (replace `{i}` with `1`, `2`, …, `dim`)

| Key | Type | Default | Description |
|---|---|---|---|
| `u0_{i}` | `CoefficientFunction` | `CF(0)` | Initial condition. |
| `b_{i}` | `Field` | `CF((0,…))` | Advection velocity (vector). |
| `d_{i}` | `Field` | `CF(0)` | Diffusion coefficient. |
| `c_{i}` | `Field` | `CF(0)` | Reaction coefficient. |
| `rhs_{i}` | `Field` | `CF(0)` | Source / right-hand side. |
| `u_bnd_{i}` | `Field` | `CF(0)` | Dirichlet boundary value. |
| `gradu_bnd_{i}` | `Field` | `CF((0,…))` | Neumann flux gradient. |
| `tot_flux_bnd_{i}` | `Field` | `CF(0)` | Total (Robin-type) flux added on the boundary as a linear-form term. |
| `mass_preserving_{i}` | `bool` | `False` | Enforce total-mass conservation. |
| `bounds_{i}` | `[lo, hi] \| None` | `None` | Enforce pointwise bounds. |

---

## Methods

### `Initialize`

Sets initial conditions, assembles the bilinear and linear forms (including the
BDF1 mass-matrix), and stores the system inverse.

---

### `PreProcess`

Snapshots `gfu_old ← gfu`.

---

### `Solve`

Updates parameter GridFunctions (evaluated at `t + dt/2` for a midpoint
approximation), reassembles, solves, and applies optional post-corrections.

---

### `PostProcess`

No-op in the primary variant (volume GridFunctions are used directly in VTK).

---

### `add_nonlinearity`

```python
def add_nonlinearity(
    target: int,
    expression: str,
    map: dict = {},
) -> None
```

Append a nonlinear source term to species `target`. See
[`ADRBoundarySystemBDF1Model.add_nonlinearity`](adr_boundary.md#add_nonlinearity)
for the full signature description — the interface is identical.

> **Note:** In the primary volume variant, nonlinear terms use `gfu_old`
> (lagged) for the species values inside the expression, so the system remains
> linear each step.
