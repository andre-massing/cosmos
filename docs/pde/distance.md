# DistanceVolumeModel

**Source:** `src/cosmos/pde/distance/distance_volume_model.py`

```python
from cosmos.pde import DistanceVolumeModel
# or the full path:
from cosmos.pde.distance.distance_volume_model import DistanceVolumeModel
```

```python
class DistanceVolumeModel(BasePDEModel)
```

Computes a smoothed distance function and its gradient on a volume compartment.

Uses a penalised Poisson solve to obtain a smooth approximation to the signed
distance from a prescribed zero-boundary, then projects the gradient direction
via an L2 projection and solves a second Poisson problem for a smoother
distance. The resulting distance field and normalised-gradient (velocity) field
are available as output for other PDE models.

The three-stage solve at each step:

1. **Penalised Poisson** — solve `(I + h² Δ) u₁ = 0` with `u₁ = 1` on the
   zero-boundary to get a smooth indicator.
2. **L2 gradient projection** — project `∇u₁ / |∇u₁|` into a vector H1 space.
3. **Poisson for distance** — solve `−Δd = div(projected gradient)` to get the
   smoothed distance `d`.

---

## Constructor

```python
DistanceVolumeModel(
    name: str = 'DistanceVolumeModel',
    model: Optional[CosmosModel] = None,
    compartment: Optional[CosmosCompartment] = None,
    **kwargs,
)
```

`is_bnd = False` / `is_vol = True` are declared as class attributes.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Identifier. |
| `model` | `CosmosModel` | Owning model. |
| `compartment` | `CosmosCompartment` | Volume compartment. |
| `zero_bnd` | `str` *(required via kwargs)* | Name of the boundary where the distance is zero (the zero level-set). |

**Raises:** `Exception` if `zero_bnd` is not provided.

---

## Parameters

| Key | Type | Default | Description |
|---|---|---|---|
| `fes_order` | `int` | `2` | Polynomial order of the H1 spaces. |
| `zero_bnd` | `str` | — | Zero-distance boundary name. |

---

## Attributes

| Attribute | Description |
|---|---|
| `gfu` | Distance GridFunction (output). |
| `gfu2` | Gradient-direction GridFunction (output). |
| `sol` | `[gfu, gfu2]` — both outputs as a list. |
| `vtk_names` | `[name + '_distance', name + '_velocity']` |

---

## Methods

### `Initialize`

Assembles the three systems (penalised-Poisson, L2 projection, Poisson) and
stores their inverses.

---

### `PreProcess`

No-op.

---

### `Solve`

Executes the three-stage pipeline each time step:

1. Re-assemble system 1; solve for the smooth indicator.
2. Re-assemble system 2; project the gradient direction.
3. Re-assemble system 3; solve for the distance.

---

### `PostProcess`

No-op.
