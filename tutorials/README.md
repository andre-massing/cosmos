# Cosmos tutorials

Worked, runnable examples demonstrating the core `cosmos` API, each validated
against a known analytical or qualitative result rather than just "run and
look plausible."

Every tutorial is a plain `.py` file using the `# %%` percent-cell format
(understood natively by VS Code, JupyterLab, and PyCharm — open it and use
"Run Cell" — or convert it to a real `.ipynb` with
[jupytext](https://jupytext.readthedocs.io/): `jupytext --to notebook <file>.py`).
They can also just be run top-to-bottom as ordinary scripts: `python3 <file>.py`.

| File | Covers |
|---|---|
| `01_heat_equation.py` | `ADRVolumeSystemBDF1Model` for pure diffusion on a unit square, Dirichlet vs. Neumann boundary conditions against known eigenfunction solutions, and a mesh-refinement convergence study. |
| `02_mean_curvature_flow.py` | `GeometricalFlowModel` + ALE driving a sphere to (near-)vanishing point under mean curvature flow, compared against the classical closed-form extinction radius, with accuracy improving under `h`/`dt` refinement. |
| `03_turing_pattern_growing_sphere.py` | Two coupled `ADRBoundarySystemBDF1Model` species forming a Turing pattern on a sphere, coupled to `GeometricalFlowModel` so local concentration drives local outward growth — a minimal proxy for pattern-driven tumor growth. |
| `04_distance_function.py` | `DistanceVolumeModel` computing a smoothed distance-to-boundary field on a disk, compared against the exact distance function under progressive mesh refinement. |

Each script prints its own validation (errors, fitted rates, comparisons
against theory) and saves at least one plot under `tutorials/figures/`.
