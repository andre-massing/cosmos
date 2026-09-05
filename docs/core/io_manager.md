# CosmosIOManager

**Source:** `src/cosmos/core/io_manager.py`

```python
class CosmosIOManager
```

Handles output for a `CosmosModel` simulation.

Creates an output directory tree under a user-specified root path, writes
VTK/VTU/PVD files for volume and boundary fields at a configurable sample rate,
and logs per-step scalar data (iteration index, time, elapsed times, and
user-defined callables) to a tab-separated text file.

### Constructor

```python
CosmosIOManager(model: CosmosModel, kwargs: dict)
```

Do not call directly; instantiated automatically by `CosmosModel.__init__`.

Reads the following keys from `kwargs`:

| Key | Type | Description |
|---|---|---|
| `root` | `str` | Root directory for output. If absent, all output is disabled. |
| `samples` | `int` | Number of evenly-spaced output snapshots. Required when `root` is set. |
| `output_callables` | `dict[str, callable]` | Extra per-step scalar quantities. Keys become column headers; values are zero-argument callables returning numbers. |

### Directory structure

```
{root}/{model.name}/
├── model_data.txt          ← mesh and compartment summary
├── output_step_data.txt    ← tab-separated step log
├── vol_pdes/               ← volume VTK output
│   └── vol_pdes_t*.vtu
└── bnd_pdes/               ← boundary VTK output
    └── bnd_pdes_t*.vtu (or *_step*.vtu for 2D curves)
```

---

### Methods

#### `initialize`

```python
def initialize() -> None
```

Create output directories, write `model_data.txt`, initialise the step-data
log, and set up `VTKOutput` writers for volume and boundary fields.

---

#### `save_step_data`

```python
def save_step_data() -> None
```

If the current time falls on a sample point, write VTK files and append one
row to the step-data log. For 2D boundary meshes, writes VTU curve files via
`write_curve_meshio` and a PVD index.

---

#### `finalize`

```python
def finalize() -> None
```

Called once after the time loop ends. Currently a no-op; override or extend for
post-processing.

---

#### `print_model_data`

```python
def print_model_data(file=None) -> None
```

Print a formatted summary of the mesh geometry, region names, element counts,
and compartment list. Writes to `file` if given, otherwise to stdout.

---

#### `print_step_data`

```python
def print_step_data(file=None) -> None
```

Print one tab-separated data row containing iteration, time, dt, step elapsed
time, ALE elapsed time, and any extra callables. Writes to `file` if given,
otherwise to stdout.

---

## Helper functions

These module-level functions are used internally by `CosmosIOManager` but are
also useful for custom output scripts.

---

### `write_curve_meshio`

```python
def write_curve_meshio(
    filename: str,
    points_xy,
    closed: bool = True,
    point_scalars: dict | None = None,
    point_vectors: dict | None = None,
    cell_scalars: dict | None = None,
    cell_vectors: dict | None = None,
) -> None
```

Write a 2D polyline curve (sequence of `(x, y)` points) to a VTU file using
the meshio library. Points are padded to 3D with `z = 0`.

| Parameter | Type | Description |
|---|---|---|
| `filename` | `str` | Output path (e.g. `"curve.vtu"`). |
| `points_xy` | `(N, 2) array-like` | 2D point coordinates. |
| `closed` | `bool` | Connect last point back to first. |
| `point_scalars` | `dict[str, (N,) array]` | Per-point scalar fields. |
| `point_vectors` | `dict[str, (N, 2\|3) array]` | Per-point vector fields. |
| `cell_scalars` | `dict[str, (M,) array]` | Per-segment scalar fields. |
| `cell_vectors` | `dict[str, (M, 2\|3) array]` | Per-segment vector fields. |

---

### `write_pvd`

```python
def write_pvd(
    out_path: str | Path,
    file_paths: Sequence[str | Path],
    timesteps: Sequence[float] | None = None,
    byte_order: str = "LittleEndian",
    version: str = "0.1",
) -> None
```

Write a `.pvd` ParaView collection file pointing to an ordered list of VTU
files, optionally annotated with simulation time values.

| Parameter | Type | Description |
|---|---|---|
| `out_path` | `str \| Path` | Output `.pvd` path. |
| `file_paths` | `Sequence` | Ordered list of file paths to include. |
| `timesteps` | `Sequence[float] \| None` | Simulation times; defaults to `0, 1, 2, …`. |
| `byte_order` | `str` | VTK byte order string. |
| `version` | `str` | VTK XML version attribute. |
