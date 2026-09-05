# ALE classes

**Source:** `src/cosmos/core/ale_manager.py`

Cosmos implements mesh motion through an Arbitrary Lagrangian–Eulerian (ALE)
framework. Three classes collaborate:

- **`CosmosALEManager`** — owned by `CosmosModel`; coordinates all ALE fields
  and owns the global displacement/velocity GridFunctions.
- **`CosmosBndALEField`** — prescribes motion on a surface compartment.
- **`CosmosVolALEField`** — prescribes motion on a volume compartment.

---

## `CosmosALEManager`

```python
class CosmosALEManager
```

Manages ALE mesh motion for a `CosmosModel`.

Maintains displacement, velocity, and reference-position GridFunctions for the
whole mesh. At each time step it collects per-compartment displacements from
registered boundary and volume ALE fields, optionally extends boundary
displacements to the bulk via a Laplace or linear-elasticity solve, and updates
the mesh deformation in NGSolve.

### Constructor

```python
CosmosALEManager(model: CosmosModel, kwargs: dict)
```

| Parameter | Type | Description |
|---|---|---|
| `model` | `CosmosModel` | Owning model. |
| `kwargs` | `dict` | Same parameter dict as passed to `CosmosModel`. Reads `volume_ALE`, `surface_ALE`, `redistribute`. |

#### `volume_ALE` options

| Value | Description |
|---|---|
| `"laplace"` *(default)* | Harmonic extension (Laplace equation on the bulk). |
| `"linel"` | Linear-elasticity extension (mesh-size–dependent Young's modulus). |

### Key attributes

| Attribute | Description |
|---|---|
| `dY` | Current displacement increment. |
| `Y` / `Yo` | Current / previous ALE position. |
| `X` / `Xo` | Current / previous material (Lagrangian) position. |
| `W` / `Wo` | Current / previous ALE mesh velocity (`dY/dt`). |
| `V` / `Vo` | Current / previous material velocity. |
| `redistribute` | Whether tangential redistribution is active. |

### Methods

#### `initialize`

```python
def initialize() -> None
```

Build `BoundaryCF` / `MaterialCF` maps from all registered ALE fields. Must be
called once before the time loop (done automatically by `CosmosModel.initialize()`).

---

#### `solve_ale`

```python
def solve_ale() -> None
```

Advance the ALE mesh by one time step:

1. Call `update()` on each registered ALE field.
2. Collect boundary and volume displacements.
3. Optionally extend boundary displacements to the bulk.
4. Update `Y`, `X`, `W`, `V`.

---

#### `finalize`

```python
def finalize() -> None
```

Commit the current deformation to the NGSolve mesh and advance `Yo`, `Xo`,
`Wo`, `Vo`. Called at the end of each time step.

---

#### `reset`

```python
def reset() -> None
```

Roll back `Y`, `X`, `W`, `V` to their previous-step values (used by the
adaptive time-stepping algorithm).

---

## `CosmosBndALEField`

```python
class CosmosBndALEField
```

ALE displacement field defined on a boundary (surface) compartment.

Prescribes the mesh displacement from a user-supplied normal velocity (and
optionally a tangential velocity) on the associated surface. Several surface
redistribution strategies are available (see `surface_ALE` in
[`CosmosModel`](model.md#keyword-parameters)).

### Constructor

```python
CosmosBndALEField(
    name: str,
    model: CosmosModel,
    compartment: CosmosCompartment,
)
```

Do not call directly; use [`CosmosModel.create_ale()`](model.md#create_ale).

### Methods

#### `set_normal_velocity`

```python
def set_normal_velocity(coef) -> None
```

Set the normal component of the mesh velocity. `coef` can be any value
accepted by [`Field`](field.md) (number, CF, GF, or callable).

---

#### `set_tangential_velocity`

```python
def set_tangential_velocity(coef) -> None
```

Set the tangential component of the mesh velocity.

---

#### `set_domain_velocity`

```python
def set_domain_velocity(coef) -> None
```

Set the full domain velocity vector directly (bypasses the
normal/tangential decomposition). When set, `set_normal_velocity` and
`set_tangential_velocity` are ignored.

---

#### `update`

```python
def update(redistribute: bool) -> None
```

Compute `ale_displ` and `mat_displ` for the current time step. Called
automatically by `CosmosALEManager.solve_ale()`.

---

## `CosmosVolALEField`

```python
class CosmosVolALEField
```

ALE displacement field defined on a volume compartment.

Prescribes the mesh motion inside a volumetric region through a user-supplied
domain velocity `CoefficientFunction`. The computed displacement is passed back
to `CosmosALEManager` each step.

### Constructor

```python
CosmosVolALEField(
    name: str,
    model: CosmosModel,
    compartment: CosmosCompartment,
)
```

Do not call directly; use [`CosmosModel.create_ale()`](model.md#create_ale).

### Methods

#### `set_domain_velocity`

```python
def set_domain_velocity(coef) -> None
```

Set the domain velocity vector field. `coef` can be any value accepted by
[`Field`](field.md).

---

#### `update`

```python
def update(redistribute: bool) -> None
```

Compute `ale_displ` and `mat_displ` for the current step.  
**Raises:** `ValueError` if `set_domain_velocity` has not been called.

