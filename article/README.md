# `article/` — simulations for the CoSMoS paper

Scripts reproducing the applications and validation tests from
["Mechanochemical Feedback between Cell Shape and Intracellular Mechanics
Revealed by a Finite-Element
Framework"](https://www.biorxiv.org/content/10.64898/2026.07.03.736361v1)
(Contri, Francis, Massing, Rangamani). Each script's own module docstring
has the full physical/biological explanation; this is just a map of what's
where.

| File | Application |
|---|---|
| [`Application_1.py`](Application_1.py) | Single-cell migration: front/back polarity reaction-diffusion system on a moving 1D membrane curve, with and without membrane-tension feedback. |
| [`Application_2.py`](Application_2.py) | Neutrophil protrusion under micropipette aspiration, coupling a diffusing protrusive signal to Willmore-flow membrane shape. |
| [`Application_3_idealized_spine.py`](Application_3_idealized_spine.py) | Dendritic-spine actin remodeling and shape change, on a synthetic idealized spine geometry. |
| [`Application_3_realistic_spine.py`](Application_3_realistic_spine.py) | The same model, on a real segmented spine geometry (see `vol/`) -- the paper's idealized-vs-realistic comparison. |
| [`dendritic_spine_geom.py`](dendritic_spine_geom.py) | Geometry generators used by `Application_3_idealized_spine.py`. |
| [`Supplementary_tests/`](Supplementary_tests/) | Solver-validation runs (pearling instability, torus evolution, stationarity, external-force robustness) supporting the applications above. |
| [`vol/`](vol/) | Segmented dendritic-spine meshes (coarse/intermediate/fine, open/closed/filled) and the scripts that repair them, used by `Application_3_realistic_spine.py` and the realistic-mesh supplementary tests. |
| [`xdmf/`](xdmf/) | HDF5/XDMF companions to the meshes in `vol/`, for visualization in ParaView. |

All scripts assume they're run with `article/` as the working directory
(relative `root`/mesh paths), and import `cosmos` from the installed
package -- see the repository root [`README.md`](../README.md) and
[`INSTALL.md`](../INSTALL.md) for setup.
