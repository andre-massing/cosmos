# `cosmos.config`

## `parameters.py` — `Config`, `get_config`, `set_config`

A single, process-wide `Config` dataclass instance holds simulation-wide settings that
would otherwise have to be threaded through every PDE model constructor:

| Field | Default | Meaning |
|---|---|---|
| `h` | `specialcf.mesh_size` | The NGSolve mesh-size symbol used in stabilization/penalty terms (e.g. interior-penalty terms in the ADR models) throughout `cosmos.pde`. |
| `seed` | `None` | Reserved for reproducible random seeding. |
| `buffer` | `2` | Number of past states kept by `SolverMesh.prev_deformation` and `SolverTime.prev_dt`/`prev_t`, i.e. how many previous time levels are available to multi-step schemes (BDF2 needs 2). |
| `precision` | `"float64"` | Reserved for numerical precision configuration. |

Access pattern:

```python
from cosmos.config.parameters import get_config, set_config

cfg = get_config()          # read the shared Config instance
set_config(buffer=3)        # mutate it in place (validates the key exists)
```

`get_config()` returns the single module-level `_current_config` instance (not a copy),
so any mutation via `set_config` is immediately visible to every object that already
holds a reference obtained from `get_config()` (e.g. every `BasePDEModel.cfg`). This
module-level singleton pattern means configuration should generally be set once, early,
before any `SolverMesh`/`SolverTime`/PDE models are constructed, since those read
`get_config().buffer` (or `.h`) at construction time.
