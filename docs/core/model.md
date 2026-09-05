# CosmosModel

**Source:** `src/cosmos/core/model.py`

```python
class CosmosModel
```

Top-level orchestrator for a time-dependent PDE simulation on an NGSolve mesh.

Collects the parent mesh together with compartments, PDE models, ALE fields,
a time manager, a step manager, and an I/O manager. The simulation is driven
step-by-step via the generator returned by `__call__()` (allows interleaving
custom logic between steps) or can be run to completion with `run()`.

---

## Constructor

```python
CosmosModel(name: str, parentmesh: Mesh, **kwargs)
```

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Human-readable identifier used as the output folder name. |
| `parentmesh` | `ngsolve.Mesh` | The underlying NGSolve mesh shared by all compartments. |
| `**kwargs` | — | Model parameters forwarded to the time, step, and I/O managers. See **Keyword parameters** below. |

### Keyword parameters

| Key | Type | Description |
|---|---|---|
| `t0` | `float` | Simulation start time. |
| `t1` | `float` | Simulation end time. |
| `dt` | `float \| list \| np.ndarray \| Parameter` | Time step (constant or per-iteration sequence). |
| `root` | `str` | Output directory root. If absent, no files are written. |
| `samples` | `int` | Number of output snapshots evenly distributed over `[t0, t1]`. Required when `root` is set. |
| `coupling_type` | `"explicit" \| "implicit"` | Coupling strategy between PDEs and ALE (default `"explicit"`). |
| `adaptive_timestep` | `bool` | Enable adaptive time-stepping (default `False`). |
| `volume_ALE` | `str` | Extension method for bulk ALE motion: `"laplace"` (default), `"linel"`. |
| `surface_ALE` | `str` | Surface redistribution method: `"mdr"` (default), `"gnz"`, `"ms"`. |
| `redistribute` | `bool` | Enable tangential mesh redistribution (default `False`). |
| `output_callables` | `dict[str, callable]` | Extra scalar quantities to append to the step-data log. |

**Raises:** `ValueError` at construction time if `**kwargs` contains any key
not listed above.

---

## Attributes

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Model name. |
| `parentmesh` | `Mesh` | NGSolve parent mesh. |
| `dim` | `int` | Ambient spatial dimension. |
| `geo_order` | `int` | Mesh geometry polynomial order. |
| `is_bnd` / `is_vol` | `bool` | Whether the mesh is a pure boundary or volume mesh. |
| `vol_ids` | `set[str]` | Set of material (codim-0) region names. |
| `bnd_ids` | `set[str]` | Set of boundary (codim-1) region names. |
| `bbnd_ids` | `set[str]` | Set of co-boundary (codim-2) region names. |
| `compartments` | `list[CosmosCompartment]` | Registered compartments. |
| `pdes` | `list[BasePDEModel]` | All registered PDE models. |
| `pdes_init` | `list[BasePDEModel]` | PDEs solved once at initialisation (ALE type `-1`). |
| `pdes_pre` | `list[BasePDEModel]` | PDEs solved before the ALE step (ALE type `0`). |
| `pdes_post` | `list[BasePDEModel]` | PDEs solved after the ALE step (ALE type `1`). |
| `ales` | `list` | Registered ALE fields. |
| `time` | `CosmosTimeManager` | Time state. |
| `step` | `CosmosStepManager` | Step solver. |
| `ale` | `CosmosALEManager` | Global ALE manager. |
| `io` | `CosmosIOManager` | I/O manager. |
| `t` | `Parameter` | NGSolve parameter for current time (alias for `time.t`). |
| `dt` | `Parameter` | NGSolve parameter for current time step (alias for `time.dt`). |

---

## Methods

### `initialize`

```python
def initialize() -> None
```

Initialise all sub-managers (time, ALE, step, I/O) and call `Initialize()` on
every registered PDE model.  Called automatically by `__call__()` and `run()`.

---

### `__call__`

```python
def __call__() -> Generator
```

Return a generator that drives the simulation one time step per `next()` call.
Useful for interleaving custom diagnostics or visualisation between steps:

```python
for step in model():
    print(f"t = {model.t.Get():.4f}")
    if some_condition:
        break
```

---

### `run`

```python
def run() -> None
```

Run the simulation to completion by exhausting the generator. Equivalent to
`for _ in model(): pass`.

---

### `set_params`

```python
def set_params(force: bool = False, **kwargs) -> None
```

Add or update entries in `model.params`.

| Parameter | Type | Description |
|---|---|---|
| `force` | `bool` | If `False` (default), raises if the key already exists. Set `True` to overwrite. |
| `**kwargs` | any | Key-value pairs to store. |

---

### `create_compartment`

```python
def create_compartment(name: str, **kwargs) -> CosmosCompartment
```

Create and register a new [`CosmosCompartment`](compartment.md).

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Unique compartment name. |
| `**kwargs` | — | Forwarded to `CosmosCompartment.__init__`. Must contain either `{material, boundary}` (volume) or `{boundary, bboundary}` (surface). |

**Returns:** the new `CosmosCompartment`.  
**Raises:** `Exception` if `name` is not unique.

---

### `create_pde`

```python
def create_pde(
    name: str,
    pde_model: type[BasePDEModel],
    compartment: CosmosCompartment,
    ale_type: int,
    **kwargs,
) -> BasePDEModel
```

Instantiate a PDE model and register it with the model.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Unique PDE name. |
| `pde_model` | `type` | A concrete subclass of `BasePDEModel`. |
| `compartment` | `CosmosCompartment` | The domain on which the PDE lives. |
| `ale_type` | `int` | Scheduling slot: `-1` (init only), `0` (before ALE), `1` (after ALE). |
| `**kwargs` | — | Extra arguments forwarded to `pde_model.__init__`. |

**Returns:** the new PDE model instance.  
**Raises:** `Exception` if the PDE/compartment dimensionality is mismatched.

---

### `create_ale`

```python
def create_ale(
    name: str,
    compartment: CosmosCompartment,
    **kwargs,
) -> CosmosBndALEField | CosmosVolALEField
```

Create and register an ALE displacement field for a compartment.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Unique ALE name. |
| `compartment` | `CosmosCompartment` | The compartment whose mesh will move. |

**Returns:** a `CosmosBndALEField` (for surface compartments) or `CosmosVolALEField` (for volume compartments).  
**Raises:** `Exception` if ALE for this domain has already been set.

---

### `print_model_data`

```python
def print_model_data() -> None
```

Print a summary of mesh and compartment data to stdout.

---

### `print_step_data`

```python
def print_step_data() -> None
```

Print the current step metrics (iteration, time, elapsed times) to stdout.
