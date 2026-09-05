# CosmosStepManager

**Source:** `src/cosmos/core/step_manager.py`

```python
class CosmosStepManager
```

Orchestrates the per-time-step solve sequence for a `CosmosModel`.

Dispatches the Initialize, PreProcess, Solve, and PostProcess phases to all
registered PDE models in the correct order (pre-ALE PDEs → ALE solve → post-ALE
PDEs). Supports explicit coupling and implicit (Gauss–Seidel iteration) coupling
modes, with optional adaptive time-stepping driven by a relative solution-change
tolerance.

---

## Constructor

```python
CosmosStepManager(model: CosmosModel, kwargs: dict)
```

Do not call directly; it is instantiated automatically by `CosmosModel.__init__`.

Reads the following keys from `kwargs`:

| Key | Type | Default | Description |
|---|---|---|---|
| `coupling_type` | `"explicit" \| "implicit"` | `"explicit"` | Coupling mode between PDEs and ALE. |
| `adaptive_timestep` | `bool` | `False` | Enable adaptive time-step control. |

---

## Attributes

| Attribute | Type | Description |
|---|---|---|
| `coupling_type` | `str` | Active coupling strategy. |
| `adaptive` | `bool` | Whether adaptive time-stepping is active. |
| `step_elasped_time` | `float \| None` | Wall-clock time (seconds) for the most recent step. |

---

## Methods

### `initialize`

```python
def initialize() -> None
```

- Calls `BasePDEModel.Initialize()` on every PDE model.
- Assembles an internal curvature-computation form used by implicit coupling.

---

### `solve_step`

```python
def solve_step() -> None
```

Advance one full time step:

1. Call `PreProcess()` on all PDEs.
2. Solve initialisation PDEs (ALE type `-1`).
3. Dispatch to `explicit_solve_step` or `implicit_solve_step_gauss` depending
   on `coupling_type` and `adaptive`.
4. Call `PostProcess()` on all PDEs.

---

### `explicit_solve_step`

```python
def explicit_solve_step() -> None
```

One step of explicit coupling:

1. Solve all pre-ALE PDEs.
2. Call `CosmosALEManager.solve_ale()`.
3. Solve all post-ALE PDEs.

---

### `implicit_solve_step_gauss`

```python
def implicit_solve_step_gauss(
    iter_max: int,
    eps_min: float,
) -> tuple[bool, float, int]
```

Gauss–Seidel iterative coupling: repeatedly solve pre-ALE PDEs, advance ALE,
solve post-ALE PDEs, until the relative change falls below `eps_min`.

| Parameter | Type | Description |
|---|---|---|
| `iter_max` | `int` | Maximum number of coupling iterations. |
| `eps_min` | `float` | Relative tolerance for convergence. |

**Returns:** `(success, eps, subiter)` — convergence flag, final relative error,
and number of iterations performed.
