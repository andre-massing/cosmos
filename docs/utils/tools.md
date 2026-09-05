# NGSolve tools

**Source:** `src/cosmos/utils/tools.py`

Low-level NGSolve helper functions.

---

## `gradient`

```python
def gradient(f, P) -> CoefficientFunction
```

Compute the projected gradient of `f` onto the plane/space defined by the
projection matrix `P`.

- For a **scalar** `f`: returns `P * (∂f/∂x, ∂f/∂y [, ∂f/∂z])`.
- For a **vector** `f`: returns the matrix whose rows are the projected
  gradients of each component (i.e. `(∇f)ᵀ` projected row-wise).

| Parameter | Type | Description |
|---|---|---|
| `f` | `CoefficientFunction` | Scalar or vector field (0 or 1 free dimensions). |
| `P` | `CoefficientFunction` | Projection matrix `(m × m)`. |

**Raises:** `RuntimeError` for tensor fields of order > 1.

---

## `MandBP` *(in `cosmos.core.utils`)*

**Source:** `src/cosmos/core/utils.py`

```python
def MandBP(
    gfu_vec,
    dt=None,
    weights=None,
    BP=None,
    MP: bool = False,
    mass0=None,
) -> np.ndarray
```

Applies bound-preserving (BP) or mass-preserving (MP) corrections to a vector.

Used internally by all ADR models after each `Solve()` call when
`bounds_{i}` or `mass_preserving_{i}` is active.

### Modes

| Mode | Condition | Behaviour |
|---|---|---|
| **Bound-preserving only** | `BP` set, `MP=False` | Clips `gfu_vec` to `[BP[0], BP[1]]`. |
| **Mass-preserving** | `MP=True` | Finds a scalar threshold `ξ` (via secant method) such that the clipped vector `clip(gfu_vec + dt·ξ, BP)` has the same weighted mass as `mass0`. |

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `gfu_vec` | `np.ndarray` | Input DOF vector to correct. |
| `dt` | `float` | Time step; required when `MP=True`. |
| `weights` | `np.ndarray` | Quadrature weights (lumped mass-matrix diagonal); required when `MP=True`. |
| `BP` | `[lo, hi]` | Bound interval; required when `BP` mode or `MP=True`. |
| `MP` | `bool` | Enable mass-preserving correction (default `False`). |
| `mass0` | `float` | Target weighted mass; required when `MP=True`. |

**Returns:** the corrected DOF vector as a `numpy.ndarray`.

### Algorithm (MP mode)

Uses `scipy.optimize.brentq` to find `ξ` such that

```
F(ξ) = Σᵢ wᵢ · clip(uᵢ + dt·ξ, lo, hi) − mass0 = 0
```

`F` is monotone in `ξ` (for `dt > 0`), so the bracket is constructed analytically:

- lower bound `ξ_lo = (lo − max(u)) / dt` → all values clip to `lo`
- upper bound `ξ_hi = (hi − min(u)) / dt` → all values clip to `hi`

Convergence tolerance is `1e-10`.
