# Mesh fixing utilities

**Source:** `src/cosmos/utils/mesh_fixing.py`

These classes allow programmatic reading, editing, and re-exporting of Netgen
`.vol` mesh files without going through the Netgen GUI.

---

## `CosmosAliasMeshSection`

```python
class CosmosAliasMeshSection
```

Represents a single named section of a Netgen mesh file.

Stores the section name, its optional header comment, the element count
(`dim`), and the list of data rows (`entries`). Provides helpers to print,
add, or replace entries, and to copy state from another section.

### Constructor

```python
CosmosAliasMeshSection(
    name: str = '',
    header: str = '',
    dim: int = 0,
    entries=None,
)
```

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Section identifier (e.g. `"surfaceelements"`, `"points"`). |
| `header` | `str` | Comment line printed before the section name in the file. |
| `dim` | `int` | Number of entries in this section. |
| `entries` | `list \| None` | Row data; each row is a list of string tokens. |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Section name. |
| `header` | `str` | Section header comment. |
| `dim` | `int` | Entry count (written as the line after the section name). |
| `entries` | `list[list[str]]` | Parsed rows, each a list of whitespace-split tokens. |

### Methods

#### `print`

```python
def print(f=None) -> None
```

Write the section in Netgen `.vol` format to file `f` (stdout if `None`).

---

#### `add_entries`

```python
def add_entries(entries) -> None
```

Append a single row `entries` to `self.entries`.

---

#### `replace_entries`

```python
def replace_entries(entries) -> None
```

Replace the entire entry list with a new one.

---

#### `set`

```python
def set(section: CosmosAliasMeshSection) -> None
```

Copy `dim` and `entries` from another `CosmosAliasMeshSection`.

---

#### `get`

```python
def get() -> tuple[int, list]
```

Return `(dim, entries)`.

---

## `CosmosAliasMesh`

```python
class CosmosAliasMesh
```

Reads, parses, and exposes a Netgen `.vol` mesh file for programmatic editing.

Loads the mesh both as an NGSolve `Mesh` object and as a collection of
`CosmosAliasMeshSection` objects, allowing individual sections (points, surface
elements, volume elements, boundary names, etc.) to be inspected and rewritten
before exporting a modified `.vol` file.

### Constructor

```python
CosmosAliasMesh(filename: str)
```

| Parameter | Type | Description |
|---|---|---|
| `filename` | `str` | Path to the `.vol` file. |

**Raises:** `Exception` if the file cannot be parsed or the mesh cannot be loaded.

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `filename` | `str` | Source file path. |
| `sections` | `dict[str, CosmosAliasMeshSection]` | All parsed sections, keyed by section name. See `SECTION_NAMES` for the full list. |
| `mesh` | `ngsolve.Mesh` | NGSolve mesh object loaded from the file. |
| `ngmesh` | `netgen.Mesh` | Underlying Netgen mesh. |

**Available section keys:** `"dimension"`, `"geomtype"`, `"facedescriptors"`,
`"surfaceelementsuv"`, `"surfaceelements"`, `"volumeelements"`,
`"edgesegmentsgi2"`, `"points"`, `"pointelements"`, `"materials"`,
`"bcnames"`, `"cd2names"`, `"cd3names"`, `"face_colours"`,
`"face_transparencies"`.

### Methods

#### `print_text`

```python
def print_text(f=None) -> None
```

Write the full mesh in Netgen `.vol` text format to `f` (stdout if `None`).

---

#### `export_mesh`

```python
def export_mesh(name: str) -> None
```

Write the mesh to `name.vol` (the `.vol` extension is appended automatically
if absent).

---

#### `get_points`

```python
def get_points() -> list[tuple[float, float, float]]
```

Return a list of `(x, y, z)` tuples for every mesh point.

---

#### `get_surface_tris`

```python
def get_surface_tris() -> list[tuple[int, int, int]]
```

Return a list of `(p1, p2, p3)` vertex-index tuples for every surface triangle.

---

#### `set_surface_tris`

```python
def set_surface_tris(tris: list[tuple[int, int, int]]) -> None
```

Overwrite the vertex indices of the surface triangles in place.

---

#### `reorient_surface_triangles_consistently`

```python
def reorient_surface_triangles_consistently(flip: bool = False) -> None
```

Walk the dual graph of the surface triangulation (via BFS) and flip triangles
so that all adjacent triangles share a consistently oriented edge.

| Parameter | Type | Description |
|---|---|---|
| `flip` | `bool` | If `True`, apply the global flip (reverse orientation). |

---

#### `globally_align_sign`

```python
def globally_align_sign(
    points: list,
    tris: list,
) -> list[tuple[int, int, int]]
```

Compute the area-weighted normal sum and the PCA third axis, then reverse all
triangle orientations if the sum and axis point in opposite directions.

---

#### `flip_normal_orientation`

```python
def flip_normal_orientation() -> None
```

Reverse the orientation of every surface triangle `(a, b, c) → (a, c, b)`.

---

#### `build_boundary_edges`

```python
def build_boundary_edges(
    tris: list[tuple[int, int, int]],
) -> list[tuple[int, int]]
```

Return the list of edges that belong to exactly one triangle (i.e. the
boundary of the surface mesh).

---

#### `build_boundary_loops`

```python
def build_boundary_loops(
    points: list,
    boundary_edges: list[tuple[int, int]],
) -> tuple[list, list]
```

Trace the boundary edges into ordered loops and compute arc-length
parameterisations.

**Returns:** `(loops, loop_edges_params)` where each loop is an ordered list of
vertex indices and each element of `loop_edges_params` is a list of
`(u, v, s0, s1)` tuples.

---

#### `fix_dim2_boundary`

```python
def fix_dim2_boundary(override: bool = False) -> None
```

Auto-populate the `edgesegmentsgi2` and `cd2names` sections by detecting
boundary loops from the surface triangulation.  Useful when generating a
surface mesh programmatically and the `.vol` file lacks edge segments.

| Parameter | Type | Description |
|---|---|---|
| `override` | `bool` | If `True`, clear any existing edge segments before rebuilding. |

**Raises:** `Exception` if edge segments already exist and `override=False`.

---

#### `mark_cd_elements`

```python
def mark_cd_elements(mesh_to_mark: CosmosAliasMesh) -> None
```

Copy boundary condition markers from `mesh_to_mark` onto `self`.

Matches surface triangles and edge segments by point coordinates (exact match
with nearest-neighbour fallback for floating-point edge cases), updates the
`bcnames` section, and remaps edge segment point indices to the new mesh.

| Parameter | Type | Description |
|---|---|---|
| `mesh_to_mark` | `CosmosAliasMesh` | Source mesh whose BC information is transferred. |

**Raises:** `Exception` if no points can be matched between the two meshes.

---

#### `build_surface_from_volume`

```python
def build_surface_from_volume(
    surfnr: int = 1,
    bcnr: int = 1,
    domin: int = 1,
    domout: int = 0,
) -> None
```

Extract boundary triangles from tetrahedral volume elements and populate the
`surfaceelements` section.  Boundary faces are those belonging to exactly one
tetrahedron.

| Parameter | Type | Description |
|---|---|---|
| `surfnr` | `int` | Face-descriptor / surface number written to each element. |
| `bcnr` | `int` | Boundary-condition number written to each element. |
| `domin` | `int` | Domain index on the interior side. |
| `domout` | `int` | Domain index on the exterior side. |

**Raises:** `Exception` if no volume elements are present or no boundary faces
are found.

---

## Standalone function: `fill_mesh`

```python
def fill_mesh(old_mesh, maxh: float) -> Mesh
```

Fill a surface mesh with tetrahedra by generating a volume mesh using Netgen.

| Parameter | Type | Description |
|---|---|---|
| `old_mesh` | `Mesh` | Input surface (boundary) mesh. |
| `maxh` | `float` | Maximum element size for the volume mesh. |

**Returns:** a new `Mesh` containing volume elements.
