# Mesh generation utilities

**Source:** `src/cosmos/utils/generate_meshes.py`

Factory functions for common geometries. Every function returns an NGSolve
`Mesh` object, curved to the requested `order_g`.

Unless stated otherwise:

- `maxh` — maximum element size.
- `order_g` — polynomial order for mesh curving (default `1`).
- Returned mesh is already curved (`mesh.Curve(order_g)` has been called).

---

## Helper

### `Meshing`

```python
def Meshing(geo, maxh: float, order_g: int, vol_or_bnd: str) -> Mesh
```

Generic meshing wrapper.

| Parameter | Type | Description |
|---|---|---|
| `geo` | Netgen geometry | Any Netgen geometry object. |
| `maxh` | `float` | Maximum element size. |
| `order_g` | `int` | Curving order. |
| `vol_or_bnd` | `"VOL" \| "BND"` | Generate a volume mesh or surface mesh only. |

---

## 1D / curve geometries

### `generate_boundary_arc`

```python
def generate_boundary_arc(
    r: float = 1,
    N: int = 20,
    bbnd_name: str = "bboundary",
) -> Mesh
```

Generate a 1D arc (half-circle) of radius `r` with `N` segments, embedded in
a 2D mesh. The two endpoints are labelled with `bbnd_name`.

---

### `generate_boundary_1D_circle`

```python
def generate_boundary_1D_circle(
    r: float = 1,
    N: int = 20,
    bbnd_name: str = "bboundary",
) -> Mesh
```

Generate a closed 1D circle of radius `r` with `N` uniformly spaced segments
in a 2D ambient mesh.

---

## 2D surface geometries (embedded in 3D)

### `generate_boundary_plane`

```python
def generate_boundary_plane(
    maxh: float = 0.1,
    order_g: int = 1,
    a: float = 1.0,
    b: float = 2.0,
    bbnd_name: str = "bboundary",
) -> Mesh
```

Rectangular flat surface of size `a × b` lying in the *z = 0* plane.
Boundary edges are labelled `bbnd_name`.

---

### `generate_boundary_circle`

```python
def generate_boundary_circle(
    maxh: float = 0.1,
    order_g: int = 1,
    R: float = 1.0,
    bbnd_name: str = "bboundary",
) -> Mesh
```

Circular disc of radius `R` embedded in 3D. Boundary edge labelled `bbnd_name`.

---

### `generate_boundary_sphere`

```python
def generate_boundary_sphere(
    maxh: float = 0.1,
    order_g: int = 1,
    center = csg.Pnt(0, 0, 0),
    R: float = 1.0,
) -> Mesh
```

Surface mesh of a sphere of radius `R` centred at `center`.

---

### `generate_boundary_ellipse`

```python
def generate_boundary_ellipse(
    maxh: float = 0.1,
    order_g: int = 1,
    a: float = 1,
    b: float = 1,
    c: float = 1,
) -> Mesh
```

Surface mesh of a triaxial ellipsoid with semi-axes `a`, `b`, `c`.

---

### `generate_boundary_cigar`

```python
def generate_boundary_cigar(
    maxh: float = 0.1,
    order_g: int = 1,
    center = csg.Pnt(0, 0, 0),
    r: float = 1.0,
    h: float = 2,
) -> Mesh
```

Surface mesh of a cylinder of radius `r` and height `h` capped with
hemispheres at both ends (a "stadium" / "capsule" shape).

---

### `generate_boundary_half_sphere`

```python
def generate_boundary_half_sphere(
    maxh: float = 0.1,
    order_g: int = 1,
    center = csg.Pnt(0, 0, 0),
    R: float = 1.0,
    bbnd_name: str = "bboundary",
) -> Mesh
```

Surface mesh of the upper hemisphere of a sphere of radius `R`. The equatorial
boundary is labelled `bbnd_name`.

---

### `generate_boundary_cylinder`

```python
def generate_boundary_cylinder(
    maxh: float = 0.1,
    R: float = 1,
    order_g: int = 1,
    bbnd_name: str = "bboundary",
) -> Mesh
```

Lateral surface of a cylinder of radius `R` (unit height, centred at origin).
The top and bottom boundary circles are labelled `bbnd_name`.

---

### `generate_boundary_torus`

```python
def generate_boundary_torus(
    maxh: float = 0.1,
    order_g: int = 1,
    center = occ.Pnt(0, 0, 0),
    R: float = 1.0,
    r: float = 0.4,
) -> Mesh
```

Surface mesh of a torus with major radius `R` and tube radius `r`.

---

### `generate_boundary_half_torus`

```python
def generate_boundary_half_torus(
    maxh: float = 0.1,
    order_g: int = 1,
    R: float = 1.0,
    r: float = 0.4,
    bbnd_name: str = "bboundary",
) -> Mesh
```

Surface mesh of a half-torus (180° revolution). Boundary edges labelled
`bbnd_name`.

---

### `generate_boundary_box`

```python
def generate_boundary_box(
    maxh: float = 0.1,
    order_g: int = 1,
    center = occ.Pnt(0, 0, 0),
    a: float = 1,
    b: float = 1,
    c: float = 1,
) -> Mesh
```

Surface mesh of a box of dimensions `a × b × c` centred at `center`.

---

### `generate_boundary_smoothed_box`

```python
def generate_boundary_smoothed_box(
    maxh: float = 0.1,
    order_g: int = 1,
    center = occ.Pnt(0, 0, 0),
    a: float = 1,
    b: float = 1,
    c: float = 1,
) -> Mesh
```

Like `generate_boundary_box` but with filleted edges (fillet radius =
`min(a, b, c) / 3`).

---

## 2D volume (filled) geometries

### `generate_volume_circle`

```python
def generate_volume_circle(
    maxh: float = 0.1,
    order_g: int = 1,
    center = occ.Pnt(0, 0, 0),
    R: float = 1.0,
    bnd_name: str = "boundary",
) -> Mesh
```

2D filled disc mesh. Boundary is labelled `bnd_name`.

---

## 3D volume geometries

### `generate_volume_ball`

```python
def generate_volume_ball(
    maxh: float = 0.1,
    order_g: int = 1,
    center = occ.Pnt(0, 0, 0),
    R: float = 1.0,
    bnd_name: str = "boundary",
) -> Mesh
```

3D ball (filled sphere) mesh. Boundary surface is labelled `bnd_name`.

---

## STL import

### `import_stl_mesh`

```python
def import_stl_mesh(
    fname: str,
    maxh: float = 0.1,
    order_g: int = 1,
) -> Mesh
```

Import and re-mesh an STL surface file using Netgen.

| Parameter | Type | Description |
|---|---|---|
| `fname` | `str` | Path to the `.stl` file. |
| `maxh` | `float` | Maximum element size. |
| `order_g` | `int` | Curving order. |
