# Geometrical flow (Willmore) models

**Sources:**
- `src/cosmos/pde/geom_flow/geometrical_flow_model.py`
- `src/cosmos/pde/geom_flow/geometrical_flow_stationary_model.py`

---

## `GeometricalFlowModel`

```python
from cosmos.pde import GeometricalFlowModel
# or the full path:
from cosmos.pde.geom_flow.geometrical_flow_model import GeometricalFlowModel
```

```python
class GeometricalFlowModel(BasePDEModel)
```

BDF1 Willmore / mean-curvature flow model on a surface compartment.

Evolves a surface by coupling a normal-velocity field *V* to the mean curvature
*κ* through a linearised Willmore-flow energy with coefficients:

| Coefficient | Param key | Physical role |
|---|---|---|
| `alpha` | `alpha` | Bending rigidity (Willmore energy weight). |
| `beta` | `beta` | Surface-tension weight. |
| `gamma` | `gamma` | Area-elasticity (penalise curvature change). |

The mixed finite-element problem couples *V* (with clamped/Navier BCs on
`compartment.clamped_bbnd` and `compartment.navier_bbnd`) and *κ*. Area- and
volume-preserving constraints are enforced by Lagrange multipliers updated via
an inner fixed-point loop.

### Constructor

```python
GeometricalFlowModel(
    name: str = 'GeometricalFlowModel',
    model: Optional[CosmosModel] = None,
    compartment: Optional[CosmosCompartment] = None,
)
```

`is_bnd = True` / `is_vol = False` are declared as class attributes.

The `compartment` must have both `clamped_bbnd` and `navier_bbnd` attributes
(set automatically when the compartment is created with those keyword
arguments).

### Parameters

| Key | Type | Default | Description |
|---|---|---|---|
| `alpha` | `CoefficientFunction` | `CF(1)` | Bending rigidity. |
| `beta` | `CoefficientFunction` | `CF(0)` | Surface-tension coefficient. |
| `gamma` | `CoefficientFunction` | `CF(0)` | Area-elasticity coefficient. |
| `sp_curv` | `CoefficientFunction` | `CF(0)` | Spontaneous (target) curvature. |
| `rhs` | `Field` | `CF(0)` | External forcing in the normal velocity equation. |
| `kappa0` | `CoefficientFunction \| None` | `None` | Override initial mean curvature (if `None`, it is computed from the mesh). |
| `area_preserving` | `bool` | `False` | Enforce area preservation. |
| `volume_preserving` | `bool` | `False` | Enforce volume preservation. |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `gfu` | `GridFunction` | Solution `(V_h, κ_h)`. |
| `V_h` | `GridFunction` | Normal velocity component. |
| `kappa_h` | `GridFunction` | Mean curvature component. |
| `vtk_names` | `list[str]` | `[name + '_V', name + '_kappa']`. |

### Methods

#### `Initialize`

Optionally computes the initial curvature from the mesh if `kappa0` is `None`.
Assembles the coupled bilinear and linear forms.

#### `PreProcess`

Updates `kappa_h_old`, normal field, Willmore tensor, and the Jacobian of the
ALE map.

#### `Solve`

Sets boundary data, assembles, and solves. If `area_preserving` or
`volume_preserving` is active, runs the inner fixed-point loop (max 10
iterations, tolerance `1e-6`) to determine the Lagrange multipliers.

#### `PostProcess`

For 2D meshes: projects `(V_h, κ_h)` into full H1 GridFunctions for VTK.

---

## `GeometricalFlowStationaryModel`

```python
from cosmos.pde import GeometricalFlowStationaryModel
# or the full path:
from cosmos.pde.geom_flow.geometrical_flow_stationary_model import GeometricalFlowStationaryModel
```

```python
class GeometricalFlowStationaryModel(BasePDEModel)
```

BDF1 geometrical flow model that co-evolves the spontaneous curvature field.

Extends `GeometricalFlowModel` by treating the spontaneous curvature *κ₀* as an
additional unknown that is advected by the ALE mesh velocity at each step. The
resulting **three-component** system *(V, κ, κ₀)* is assembled and solved as a
single linear system each time step.

This model is appropriate when the spontaneous curvature depends on a separate
chemical field (e.g. a membrane-bound protein concentration) and must be
transported consistently with the evolving surface.

### Constructor

```python
GeometricalFlowStationaryModel(
    name: str = 'GeometricalFlowStationaryModel',
    model: Optional[CosmosModel] = None,
    compartment: Optional[CosmosCompartment] = None,
)
```

`is_bnd = True` / `is_vol = False` are declared as class attributes.

### Parameters

Same as `GeometricalFlowModel`, except:

- `sp_curv` is **not** a fixed parameter; the spontaneous curvature is now a
  dynamical unknown `sp_curv_h` initialised by `kappa0` (or from the initial
  mesh curvature if `kappa0 is None`).

### Attributes

| Attribute | Description |
|---|---|
| `V_h` | Normal velocity. |
| `kappa_h` | Mean curvature. |
| `sp_curv_h` | Spontaneous curvature (co-evolved unknown). |
| `vtk_names` | `[name + '_V', name + '_kappa', name + '_kappa0']`. |

### Methods

Same lifecycle as `GeometricalFlowModel`. `PreProcess` additionally snapshots
`sp_curv_h_old ← sp_curv_h` before each step.
