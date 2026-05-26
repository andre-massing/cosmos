# Cosmos — API Reference

Cosmos is a finite-element PDE solver built on top of [NGSolve](https://ngsolve.org/). It couples arbitrary numbers of advection–diffusion–reaction systems (on volumes and surfaces), geometrical flows, and an Arbitrary Lagrangian–Eulerian (ALE) mesh-motion framework into a single time-stepping loop.

---

## Package layout

| Module | Contents |
|---|---|
| [`cosmos.config`](config.md) | Global solver configuration (`Config`, `get_config`, `set_config`) |
| [`cosmos.core`](core/index.md) | Core runtime: model, compartments, fields, ALE, time and step managers |
| [`cosmos.pde`](pde/index.md) | PDE model classes (ADR, distance, Willmore flow) |
| [`cosmos.io`](io.md) | I/O manager, VTK/PVD writers |
| [`cosmos.utils`](utils/index.md) | Mesh generation, mesh fixing, manufactured-solution tools |

---

## Quick-start sketch

```python
from ngsolve import *
from cosmos.core.model import CosmosModel
from cosmos.pde.adr.boundary.adr_boundary_system_bdf1_model_nostab import ADRBoundarySystemBDF1Model

mesh = Mesh(...)  # any NGSolve Mesh

model = CosmosModel("my_sim", mesh,
                    t0=0, t1=1, dt=0.01,
                    root="output", samples=10)

surf = model.create_compartment("surface",
                                boundary="membrane",
                                bboundary="")

ale  = model.create_ale("surface_ale", surf)
ale.set_normal_velocity(CF(0))
ale.set_tangential_velocity(CF((0,)*mesh.dim))

adr = model.create_pde("species", ADRBoundarySystemBDF1Model, surf,
                       ale_type=1, dim=1)
adr.set_params(d_1=CF(1e-3), c_1=CF(1), u0_1=CF(1))

model.run()
```

---

## Design overview

```
CosmosModel
├── compartments []       CosmosCompartment
├── pdes []               BasePDEModel  (concrete subclass)
├── ales []               CosmosBndALEField | CosmosVolALEField
├── time                  CosmosTimeManager
├── step                  CosmosStepManager
├── ale                   CosmosALEManager
└── io                    CosmosIOManager
```

The `CosmosModel.__call__()` method returns a generator; each `next()` call
advances one time step.  `model.run()` exhausts the generator in a single
blocking call.
