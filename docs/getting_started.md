# Getting Started

## Prerequisites

CoSMoS is built on top of [NGSolve](https://ngsolve.org/) (which bundles the `netgen`
mesh generator) and requires Python 3.10+. NGSolve is not installable from PyPI as a
plain dependency for every platform; the project builds it from source. See
`reproduction/howto_install/install.txt` for the reference build recipe (clone NGSolve,
build with CMake, add the resulting `inst/bin` to `PATH` and the NGSolve Python bindings
to `PYTHONPATH`).

Once NGSolve is available, install the rest of the scientific-Python dependencies and the
package itself. The repository ships a `Pipfile` with the runtime dependencies
(`numpy`, `scipy`, `pandas`, `matplotlib`, `meshio`, `pytest`, ...) and declares `cosmos`
itself as an editable local package:

```bash
pipenv install
```

Equivalently, with `pip` inside an environment that already has NGSolve on the
`PYTHONPATH`:

```bash
pip install -e .
pip install numpy scipy pandas matplotlib meshio pytest
```

The package metadata lives in `pyproject.toml` (build backend: `hatchling`, package
sources under `src/`). Optional extras are declared there too:

```bash
pip install -e ".[tests]"   # pytest
pip install -e ".[docs]"    # sphinx
```

## Running the test suite

```bash
pytest
```

`pyproject.toml` registers custom markers used across `tests/`: `slow`, `visual`,
`convergence`, `physical`. For example, to skip long-running simulations:

```bash
pytest -m "not slow"
```

## A minimal example

The pattern below mirrors what the test suite under `tests/pde/physical` does: build a
mesh, wrap it, create a `Solver`, attach one PDE model, and step the simulation. This
example solves a boundary (surface) advection-diffusion-reaction problem on a sphere.

```python
from ngsolve import *
from cosmos import (
    Solver, SolverMesh, SolverTime, ADRBoundaryBDF1Model,
)
from cosmos.utils.generate_meshes import generate_boundary_sphere

# 1. Build the mesh and wrap it
mesh = generate_boundary_sphere(maxh=0.3, order_g=2)
solvermesh = SolverMesh(mesh)

# 2. Configure time stepping
solvertime = SolverTime(dt=1e-3, initial_t=0.0, final_t=0.1)

# 3. Create the solver
solver = Solver(solvermesh, solvertime, name="adr_demo", printing=True)

# 4. Attach a PDE model (model_order controls solve order within a step)
adr = ADRBoundaryBDF1Model(
    solver,
    model_order=1,
    input_params={"u0": CF(1), "fes_order": 1},
)
adr.set_input_fields({"d": CF(0.1)})  # diffusion coefficient

# 5. (Optional) configure periodic VTK output
solver.output_params(folder="results", sample_rate=10)
solver.save_model_solution(adr)

# 6. Run to completion
solver.run()

print("Final solution field:", adr.sol)
```

To couple a shape-evolution model (e.g. Willmore flow) with mesh motion, attach an
`ALEModel` at a later `model_order` and prescribe its boundary displacement from the
shape model's output — see the pattern used in
`tests/pde/physical/willmore/test_helfrich_spine.py` and
[coupling.md](coupling.md).

## Where to look next

- [architecture.md](architecture.md) explains what happens on every simulation step.
- [pde_models.md](pde_models.md) lists every available PDE model and its parameters.
- [utils.md](utils.md) documents the mesh-generation helpers used above.
