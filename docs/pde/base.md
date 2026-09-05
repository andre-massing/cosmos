# BasePDEModel

**Source:** `src/cosmos/pde/base.py`

```python
from abc import ABC, abstractmethod
class BasePDEModel(ABC)
```

Abstract base class for all PDE models in Cosmos.

Defines the four-phase lifecycle every concrete model must implement:

| Phase | Method | Purpose |
|---|---|---|
| 1 | `Initialize()` | Set up FE spaces; assemble time-independent forms. |
| 2 | `PreProcess()` | Snapshot the solution before the current solve. |
| 3 | `Solve()` | Assemble and invert the linear system. |
| 4 | `PostProcess()` | Project results for output; apply post-corrections (mass / bound preservation). |

---

## Constructor

```python
BasePDEModel(
    name: str = 'BasePDEModel',
    model: Optional[CosmosModel] = None,
    compartment: Optional[CosmosCompartment] = None,
)
```

Do not instantiate directly; subclass and pass to
[`CosmosModel.create_pde()`](../core/model.md#create_pde).

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Human-readable identifier. |
| `model` | `CosmosModel` | Owning model. |
| `compartment` | `CosmosCompartment` | Domain on which the PDE is defined. |

---

## Attributes

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | PDE model name. |
| `model` | `CosmosModel` | Owning model. |
| `compartment` | `CosmosCompartment` | Associated compartment. |
| `params` | `dict` | Dictionary of named parameters (access via `set_params`). |
| `cfg` | `Config` | Global solver configuration (from `get_config()`). |
| `vtk_gfu` | `list[GridFunction]` | GridFunctions written to VTK when `printing=True`. |
| `vtk_names` | `list[str]` | Names of the VTK fields. |

`is_bnd`/`is_vol` are **class attributes**, not instance attributes — each
concrete subclass declares them once at the top of the class body (e.g.
`is_bnd = True; is_vol = False` for a surface model), since whether a given
PDE type lives on a boundary or a volume never varies between instances of
that class. `BasePDEModel` itself does not set a default for either.

---

## Abstract methods

All four methods **must** be overridden in every concrete subclass.

### `Initialize`

```python
@abstractmethod
def Initialize() -> None
```

Set up finite-element spaces, assemble bilinear/linear forms that do not
change between time steps, and compute initial conditions.

---

### `PreProcess`

```python
@abstractmethod
def PreProcess() -> None
```

Store the solution from the previous time step (typically
`gfu_old.vec.data = gfu.vec.data`) and perform any other setup needed before
`Solve()`.

---

### `Solve`

```python
@abstractmethod
def Solve() -> None
```

Assemble and solve the linear system for the current time step. Update `gfu`.

---

### `PostProcess`

```python
@abstractmethod
def PostProcess() -> None
```

After `Solve()`: project the solution for 2D VTK output, apply mass/bound-
preserving post-processing, or perform any other cleanup.

---

## Concrete methods

### `set_params`

```python
def set_params(**kwargs) -> None
```

Update entries in `self.params`.

- If the existing value is a [`Field`](../core/field.md), the field's `.cf`
  setter is called (the underlying GridFunction is updated in place).
- Otherwise the dictionary entry is replaced directly.

**Raises:** `ValueError` if `key` is not already present in `params`.

---

### `adaptive_timestep_cap`

```python
def adaptive_timestep_cap() -> bool
```

Hook for adaptive time-stepping. Override to return `True` when the model
requests the current time step to be capped. Default returns `False`.

---

### `reset`

```python
def reset() -> None
```

Roll back the PDE state to the start of the current step (used by adaptive
time-stepping). Default is a no-op; override as needed.
