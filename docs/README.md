# CoSMoS Documentation

CoSMoS (**Co**ntinuum **S**urface **M**echanics **S**imulator) is a Python package built on
top of [NGSolve](https://ngsolve.org/) that provides finite-element solvers for partial
differential equations (PDEs) posed on deforming membranes and bulk domains. It targets
problems from biological membrane mechanics: reaction-diffusion processes, phase separation
(Cahn-Hilliard), and curvature-driven flows (mean curvature flow, Willmore/Helfrich flow),
optionally coupled to an Arbitrary Lagrangian-Eulerian (ALE) mesh motion model.

This `docs/` folder contains the package documentation. It is organized as follows:

| Document | Contents |
|---|---|
| [architecture.md](architecture.md) | High-level design, the core abstractions and how a simulation runs end to end |
| [getting_started.md](getting_started.md) | Installation and a minimal worked example |
| [core.md](core.md) | The `cosmos.core` subpackage: mesh, time, solver, iterator, fields |
| [config.md](config.md) | The `cosmos.config` subpackage: global simulation configuration |
| [pde_models.md](pde_models.md) | The `cosmos.pde` subpackage: base model contract and all PDE model families |
| [coupling.md](coupling.md) | The `cosmos.coupling` subpackage: ALE mesh-motion coupling |
| [utils.md](utils.md) | The `cosmos.utils` subpackage: mesh generation and manufactured-solution helpers |

## Package layout

```
src/cosmos/
├── __init__.py          # public API re-exports
├── config/               # global Config object (mesh-size symbol, buffer size, ...)
├── core/                 # mesh/time/solver/field/iterator building blocks
├── coupling/              # ALE mesh-motion model
├── io/                    # logging setup, (placeholder) visualization helpers
├── pde/                   # PDE model implementations, one subpackage per physics
│   ├── adr/                # advection-diffusion-reaction (volume & boundary)
│   ├── cahn_hilliard/       # Cahn-Hilliard phase separation (volume & boundary)
│   ├── mean_curvature/       # mean curvature flow (boundary only)
│   └── willmore/             # Willmore / Helfrich bending flow (boundary only)
└── utils/                 # mesh generation and manufactured-solution tooling
```

See [architecture.md](architecture.md) for how these pieces fit together at runtime.
