# cosmos.config

**Source:** `src/cosmos/config/parameters.py`

Global configuration dataclass and accessor functions shared across the entire solver.

---

## `Config`

```python
@dataclass
class Config
```

Global configuration dataclass for the Cosmos solver.

Stores shared numerical settings (mesh-size reference `h`, floating-point
precision, buffer size, and an optional random seed) that are read by PDE
models via [`get_config()`](#get_config).

### Attributes

| Name | Type | Default | Description |
|---|---|---|---|
| `h` | `CoefficientFunction` | `specialcf.mesh_size` | NGSolve mesh-size coefficient, used inside stabilisation terms. |
| `seed` | `Optional[int]` | `None` | Optional random seed for reproducibility. |
| `buffer` | `int` | `2` | Internal buffer size used by some assembly routines. |
| `precision` | `str` | `"float64"` | Floating-point precision string. |

---

## `get_config`

```python
def get_config() -> Config
```

Return the current process-wide `Config` object (read-only use intended).

**Returns:** the singleton `Config` instance.

---

## `set_config`

```python
def set_config(**kwargs) -> None
```

Update one or more fields of the process-wide `Config`.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `**kwargs` | any | Key-value pairs matching attributes of `Config`. |

**Raises:** `ValueError` if a key does not match an existing `Config` attribute.

**Example:**

```python
from cosmos.config.parameters import set_config
set_config(precision="float32", seed=42)
```
