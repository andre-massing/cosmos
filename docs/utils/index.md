# cosmos.utils

Utility modules that are independent of the simulation runtime.

| Module | Contents |
|---|---|
| [`mesh_fixing`](mesh_fixing.md) | `CosmosAliasMeshSection`, `CosmosAliasMesh`, `fill_mesh` — read, edit, and re-export Netgen `.vol` mesh files |
| [`generate_meshes`](generate_meshes.md) | Factory functions for common geometry types (sphere, cylinder, torus, box, …) |
| [`tools`](tools.md) | NGSolve helper: projected gradient |
| `core/utils` | `MandBP` — mass and bound-preserving post-processing (used internally by ADR models) |

All of the above (except `core/utils`) are also re-exported directly from
`cosmos.utils`, e.g. `from cosmos.utils import generate_boundary_sphere`.
