# cosmos.core

The `core` subpackage provides the runtime machinery of every Cosmos simulation.

| Module | Key classes |
|---|---|
| [`model`](model.md) | `CosmosModel` — top-level simulation orchestrator |
| [`compartment`](compartment.md) | `CosmosCompartment` — named sub-domain (volume or surface) |
| [`field`](field.md) | `Field`, `InputField`, `OutputField` — coefficient wrappers |
| [`ale_manager`](ale_manager.md) | `CosmosALEManager`, `CosmosBndALEField`, `CosmosVolALEField` — mesh motion |
| [`step_manager`](step_manager.md) | `CosmosStepManager` — per-step solve dispatcher |
| [`time_manager`](time_manager.md) | `CosmosTimeManager`, `CosmosTimeHelper` — time state and advancement |
| [`io_manager`](io_manager.md) | `CosmosIOManager` — VTK/step-log output |
| `utils` | `MandBP` — mass and bound-preserving post-processing |

---

## Lifecycle of a simulation

```
CosmosModel.initialize()
│
│   ← CosmosTimeManager.initialize()
│   ← CosmosALEManager.initialize()
│   ← CosmosStepManager.initialize()
│       └── BasePDEModel.Initialize()  for each pde
│   ← CosmosIOManager.initialize()
│
│   io.save_step_data()   ← write t=0 snapshot
│
└─ while t <= t1:
       CosmosStepManager.solve_step()
       │   ← PreProcess()  for each pde
       │   ← Solve()       for pdes_pre
       │   ← CosmosALEManager.solve_ale()
       │   ← Solve()       for pdes_post
       │   ← PostProcess() for each pde
       CosmosTimeManager.next()
       CosmosIOManager.save_step_data()
       CosmosALEManager.finalize()
```

Every manager (`CosmosALEManager`, `CosmosStepManager`, `CosmosIOManager`) is
constructed with a back-reference to its owning `CosmosModel` (stored as
`self.model`), so none of its methods need `model` passed in again.
