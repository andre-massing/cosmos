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
| `utils` | `MandBP` — mass and bound-preserving post-processing |

---

## Lifecycle of a simulation

```
CosmosModel.initialize()
│
│   ← CosmosTimeManager.initialize()
│   ← CosmosALEManager.initialize(model)
│   ← CosmosStepManager.initialize(model)
│       └── BasePDEModel.Initialize()  for each pde
│   ← CosmosIOManager.initialize(model)
│
│   io.save_step_data(model)   ← write t=0 snapshot
│
└─ while t <= t1:
       CosmosStepManager.solve_step(model)
       │   ← PreProcess()  for each pde
       │   ← Solve()       for pdes_pre
       │   ← CosmosALEManager.solve_ale(model)
       │   ← Solve()       for pdes_post
       │   ← PostProcess() for each pde
       CosmosTimeManager.next()
       CosmosIOManager.save_step_data(model)
       CosmosALEManager.finalize(model)
```
