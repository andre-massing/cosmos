# CosmosCompartment

**Source:** `src/cosmos/core/compartment.py`

```python
class CosmosCompartment
```

Represents a computational sub-domain (compartment) within a
[`CosmosModel`](model.md).

A compartment wraps either a **volume domain**—identified by a material name and
its bounding surface—or a **surface domain**—identified by a boundary name and
its co-dimension-2 boundary. It holds a list of PDE models assigned to it and
an optional ALE field.

---

## Constructor

```python
CosmosCompartment(
    name: str = '',
    model: CosmosModel = None,
    **kwargs,
)
```

Do not call this constructor directly; use
[`CosmosModel.create_compartment()`](model.md#create_compartment) instead.

### Volume compartment

Pass both `material` and `boundary`:

```python
comp = model.create_compartment(
    "cytoplasm",
    material="vol",
    boundary="membrane",
)
```

| kwarg | Type | Description |
|---|---|---|
| `material` | `str` | Pipe-separated NGSolve material name(s) (e.g. `"vol"` or `"vol1|vol2"`). Must match `model.vol_ids`. |
| `boundary` | `str` | Pipe-separated boundary name(s) enclosing the material. Must match `model.bnd_ids`. |

### Surface compartment

Pass both `boundary` and `bboundary`:

```python
comp = model.create_compartment(
    "membrane",
    boundary="membrane",
    bboundary="edge",
)
```

| kwarg | Type | Description |
|---|---|---|
| `boundary` | `str` | Pipe-separated boundary name(s). Must match `model.bnd_ids`. |
| `bboundary` | `str` | Pipe-separated co-boundary name(s) (can be `""`). Must match `model.bbnd_ids`. |
| `clamped_bbnd` | `str` | *(optional)* Co-boundary names with clamped (fully fixed) conditions. |
| `navier_bbnd` | `str` | *(optional)* Co-boundary names with Navier (free-sliding) conditions. |

---

## Attributes

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Compartment name. |
| `model` | `CosmosModel` | Parent model. |
| `dim` | `int` | Spatial dimension of this compartment (= `model.dim` for volumes, `model.dim - 1` for surfaces). |
| `dim_emd` | `int` | Ambient (embedding) dimension (always `model.dim`). |
| `is_vol` | `bool` | `True` for a volume compartment. |
| `is_bnd` | `bool` | `True` for a surface compartment. |
| `domain_id` | `str` | Material/boundary name string as supplied. |
| `boundary_id` | `str` | Boundary/co-boundary name string as supplied. |
| `domain` | `ngsolve region` | NGSolve region object for the interior. |
| `boundary` | `ngsolve region` | NGSolve region object for the boundary. |
| `pdes` | `list[BasePDEModel]` | PDE models operating on this compartment. |
| `ale` | `CosmosBndALEField \| CosmosVolALEField \| None` | ALE field registered on this compartment. |
| `clamped_bbnd` | `str` | *(surface only)* Clamped co-boundary name. |
| `navier_bbnd` | `str` | *(surface only)* Navier co-boundary name. |

---

## Methods

### `print_compartment_info`

```python
def print_compartment_info() -> None
```

Print a human-readable summary of the compartment (name, dimension, domain
name, boundary name) to stdout.
