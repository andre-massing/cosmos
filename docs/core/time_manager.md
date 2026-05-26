# Time managers

**Source:** `src/cosmos/core/time_manager.py`

Two classes manage simulation time: `CosmosTimeManager` (the public API
consumed by `CosmosModel`) and `CosmosTimeHelper` (the parsing and advancement
logic created at initialisation time).

---

## `CosmosTimeManager`

```python
class CosmosTimeManager
```

Manages the simulation time state (current time, time step, iteration counter).

Exposes NGSolve `Parameter` objects for `t` and `dt` so they can be embedded
directly inside coefficient functions and weak forms. Delegates
time-advancement logic to a `CosmosTimeHelper` that is created when
`initialize()` is called.

### Constructor

```python
CosmosTimeManager(kwargs: dict)
```

Do not call directly; instantiated automatically by `CosmosModel.__init__`.

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `t` | `Parameter` | Current simulation time (NGSolve `Parameter`, usable in CFs). |
| `dt` | `Parameter` | Current time step size. |
| `iter` | `int` | Number of completed time steps. |
| `t0` | `float` | Start time (set after `initialize()`). |
| `t1` | `float` | End time (set after `initialize()`). |
| `dt0` | `float` | Initial time step size (set after `initialize()`). |
| `prev_t` | `list[float]` | Ring buffer of the last ≤ 6 time values. |
| `prev_dt` | `list[float]` | Ring buffer of the last ≤ 6 time-step sizes. |
| `helper` | `CosmosTimeHelper` | Underlying helper (set after `initialize()`). |

### Methods

#### `initialize`

```python
def initialize() -> None
```

Create the `CosmosTimeHelper` from the model parameters and set the initial
values of `t` and `dt`.

---

#### `next`

```python
def next() -> None
```

Advance the simulation by one time step: increment `iter`, call
`helper.next()` to update `t` and `dt`, and append to the ring buffers.

---

#### `modify_dt`

```python
def modify_dt(dt_new: float) -> None
```

Override the current time step with `dt_new`. Also updates the last entry in
`prev_dt`. Used by the adaptive time-stepping algorithm in `CosmosStepManager`.

---

## `CosmosTimeHelper`

```python
class CosmosTimeHelper
```

Parses time-discretisation parameters and implements step advancement.

Accepts scalar, list, or `numpy.ndarray` time-step specifications; when a
sequence is provided the time step is updated from the sequence at each
iteration. Validates that `t0`, `t1`, and `dt` are present and that `dt` is
strictly positive.

### Constructor

```python
CosmosTimeHelper(**kwargs)
```

Required keyword arguments:

| Key | Type | Description |
|---|---|---|
| `t0` | `float` | Simulation start time. |
| `t1` | `float` | Simulation end time. |
| `dt` | `float \| list \| np.ndarray \| Parameter` | Constant step size, or a sequence of per-step sizes. |

Optional:

| Key | Type | Description |
|---|---|---|
| `t` | `Parameter` | If supplied, this parameter is kept in sync with the current time. |

**Raises:** `Exception` if any required key is missing, has the wrong type, or
`dt` ≤ 0.

### Methods

#### `initialize`

```python
def initialize(t: Parameter, dt: Parameter) -> None
```

Set `t` and `dt` to their initial values (`t0` and the first step size).

---

#### `next`

```python
def next(iter: int, t: Parameter, dt: Parameter) -> None
```

Advance time: `t ← t + dt`, then update `dt` from the sequence (if a list or
array was provided) or keep it constant.

| Parameter | Type | Description |
|---|---|---|
| `iter` | `int` | Current iteration index, used to index a list/array `dt`. |
| `t` | `Parameter` | Time parameter to advance. |
| `dt` | `Parameter` | Time-step parameter to update. |

**Raises:** `Exception` if the new `dt` is ≤ 0.
