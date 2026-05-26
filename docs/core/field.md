# Field class

**Source:** `src/cosmos/core/field.py`

The `Field` class provides a uniform interface for coefficient functions used as
PDE parameters. PDE models write against `Field` without knowing whether the
underlying data is a constant, an expression, or a computed `GridFunction`.

---

## `Field`

```python
class Field
```

Uniform wrapper around an NGSolve coefficient or callable.

Accepts a plain number, a `CoefficientFunction`, a `GridFunction`, or any
zero-argument callable that returns a `CoefficientFunction`, and exposes them
through a consistent `__call__` / `.cf` interface so that PDE parameter fields
can be changed transparently at run time.

### Constructor

```python
Field(coef: int | float | CoefficientFunction | GridFunction | Callable)
```

| Parameter | Type | Description |
|---|---|---|
| `coef` | `int \| float \| CoefficientFunction \| GridFunction \| Callable` | The underlying coefficient. A plain number is wrapped in `CF(coef)`. |

**Raises:** `TypeError` if `coef` is none of the accepted types.

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `is_callable` | `bool` | `True` when `coef` was supplied as a callable. |
| `cf` | property | Evaluates and returns the `CoefficientFunction`. Settable: assigns a new coefficient. |

### Methods

#### `__call__`

```python
def __call__() -> CoefficientFunction
```

Evaluate and return the underlying `CoefficientFunction`.

#### `cf` (property setter)

```python
field.cf = new_coef
```

Replace the stored coefficient with `new_coef` (accepts number,
`CoefficientFunction`, `GridFunction`, or callable).
