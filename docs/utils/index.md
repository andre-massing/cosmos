# cosmos.utils

Utility modules that are independent of the simulation runtime.

| Module | Contents |
|---|---|
| [`mesh_fixing`](mesh_fixing.md) | `CosmosAliasMeshSection`, `CosmosAliasMesh` — read, edit, and re-export Netgen `.vol` mesh files |
| [`generate_meshes`](generate_meshes.md) | Factory functions for common geometry types (sphere, cylinder, torus, box, …) |
| [`dendritic_spine_geom`](generate_meshes.md#dendritic-spine-geometries) | `generate_synapse2d`, `generate_synapse3d` — dendritic spine meshes |
| [`manufactured_solution_tools`](manufactured_solution_tools.md) | SymPy-based tools for computing manufactured solutions (Laplace-Beltrami RHS, normal fields, …) |
| [`tools`](tools.md) | NGSolve helpers: projected gradient, stabilised mean curvature |
| `core/utils` | `MandBP` — mass and bound-preserving post-processing (used internally by ADR models) |
