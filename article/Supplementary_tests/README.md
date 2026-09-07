# `article/Supplementary_tests/`

Solver-validation runs supporting the applications in `../`: each checks a
classical geometric-flow behavior or a robustness property, rather than a
biological result. See each script's own module docstring for the full
explanation.

| Script | Checks |
|---|---|
| [`test_pearling.py`](test_pearling.py) | The pearling/beading instability of a membrane tube reproduces correctly, and bending energy decreases monotonically throughout (the flow is a genuine gradient flow). |
| [`test_torus_evolution.py`](test_torus_evolution.py) | Volume-preserving Willmore flow of a (non-Clifford-ratio) torus also decreases bending energy, exercising a genus-1 surface and the volume-preservation constraint together. |
| [`test_stationarity.py`](test_stationarity.py) | A real, segmented spine mesh at its own resting curvature (no forcing, no curvature mismatch) stays stationary -- a self-consistency check on irregular real geometry, run on both open and closed mesh variants. |
| [`test_external_force.py`](test_external_force.py) | A real, segmented spine mesh remains numerically stable under a transient, spatially-localized external force. |

All four load meshes from [`../vol/`](../vol/) or generate idealized
geometries via `cosmos.utils.generate_meshes`, and use
`GeometricalFlowModel`/`GeometricalFlowStationaryModel` exactly as the
`Application_*.py` scripts do.
