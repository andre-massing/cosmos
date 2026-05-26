# Manufactured solution tools

**Source:** `src/cosmos/utils/manufactured_solution_tools.py`

SymPy-based utilities for deriving manufactured solutions of surface PDEs.
Requires SymPy and IPython/Jupyter (for display functions).

All functions operate on SymPy symbolic expressions.

---

## Differential operators

### `Grad`

```python
def Grad(f, x) -> Matrix
```

Compute the gradient of scalar `f` or column-vector `f` with respect to
variables `x`.

| Parameter | Type | Description |
|---|---|---|
| `f` | SymPy scalar or `(n, 1)` Matrix | Function to differentiate. |
| `x` | SymPy Matrix / list of symbols | Variables. |

**Raises:** `RuntimeError` for unsupported shapes.

---

### `Project_Grad`

```python
def Project_Grad(f, x, n) -> Matrix
```

Compute the tangential (surface) gradient by projecting `Grad(f, x)` onto the
plane perpendicular to `n`:

```
∇_Γ f = ∇f − (n · ∇f) n
```

---

### `Div`

```python
def Div(f, x) -> expr
```

Compute the divergence of a vector field or the row-wise divergence of a
matrix field. `"Compute divergence."`

---

### `Laplace`

```python
def Laplace(f, x) -> expr
```

Compute the (full) Laplacian of scalar `f`. `"Compute the (full) Laplacian."`

---

### `Hess`

```python
def Hess(f, x) -> Matrix
```

Compute the Hessian matrix of scalar `f`. `"Compute the Hessian of f."`

---

### `Tr`

```python
def Tr(f) -> expr
```

Compute the trace of a square matrix. `"Compute trace."`

---

### `LaplaceBeltrami`

```python
def LaplaceBeltrami(f, x, n) -> expr
```

Compute the Laplace-Beltrami operator of scalar `f` on the surface defined by
normal `n`:

```
Δ_Γ f = Δf − n · (∇²f n) − tr(∇n) ∇f · n
```

`"Compute Laplace-Beltrami operator of u."`

---

### `compute_normal`

```python
def compute_normal(phi, xx) -> Matrix
```

Compute the unit outward normal of the level-set surface `{φ = 0}`:

```
n = ∇φ / |∇φ|
```

---

### `vec_norm`

```python
def vec_norm(xx) -> expr
```

Compute the Euclidean norm of a list of SymPy expressions.

---

## Display utilities

### `vec_simplify`

```python
def vec_simplify(f) -> Matrix
```

Apply `simplify` component-wise to a SymPy Matrix. `"Simplify vector expression."`

---

### `eprint`

```python
def eprint(e, name=None, maxsize=500) -> None
```

Pretty-print a SymPy expression in a Jupyter notebook.

Uses LaTeX/MathJax for small expressions and SymPy unicode for large ones,
as determined by the operation count compared to `maxsize`.

| Parameter | Type | Description |
|---|---|---|
| `e` | SymPy expr or Matrix | Expression to display. |
| `name` | `str \| None` | If given, prepended as `"name = expr"`. |
| `maxsize` | `int` | Operation-count threshold; above this, unicode is used instead of LaTeX. |

---

## Manufactured-solution helper

### `get_solution_str`

```python
def get_solution_str(
    u_str: str,
    levelset_str: str,
    epsilon_str: str = "1",
    b_str: list[str] = ["0", "0", "0"],
    c_str: str = "1",
) -> tuple[str, list[str]]
```

Given an analytical reference solution and a level-set description of a
surface, compute the right-hand side and tangential gradient for the problem

```
−ε Δ_Γ u + c u + b · ∇_Γ u = f   on Γ = {φ = 0}
```

All arguments are Python-parseable string expressions in `x`, `y`, `z`.

| Parameter | Type | Description |
|---|---|---|
| `u_str` | `str` | Manufactured solution expression (e.g. `"sin(x)*cos(y)"`). |
| `levelset_str` | `str` | Level-set function `φ` (e.g. `"x**2 + y**2 + z**2 - 1"` for the unit sphere). |
| `epsilon_str` | `str` | Diffusion coefficient (default `"1"`). |
| `b_str` | `list[str]` | Advection field components (default zero). |
| `c_str` | `str` | Reaction coefficient (default `"1"`). |

**Returns:** `(f_str, tang_grad_u_str)` — string representations of the RHS
and each component of `∇_Γ u`, ready to be converted to NGSolve coefficient
functions with `eval`.

**Example:**

```python
from cosmos.utils.manufactured_solution_tools import get_solution_str

f_str, grad_str = get_solution_str(
    u_str="sin(pi*x/2)*sin(pi*y/2)*sin(pi*z/2)",
    levelset_str="x**2 + y**2 + z**2 - 1",
)
```
