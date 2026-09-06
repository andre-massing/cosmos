# cosmos

Cosmos, the Continuum Surface Mechanics Simulator, provides a number of finite element based
solvers for the numerical solution of PDE problems related to cell membrane dynamics: coupled
advection-diffusion-reaction systems on volumes and surfaces, geometrical (Willmore /
mean-curvature) flow, and an Arbitrary Lagrangian-Eulerian (ALE) framework for moving-domain
and moving-surface problems, all built on top of [NGSolve](https://ngsolve.org/).

## Getting started

- **Installing:** see [`INSTALL.md`](INSTALL.md) for a full, reproducible setup
  (Python version, NGSolve/Netgen, and pinned Python dependencies).
- **API reference:** see [`docs/index.md`](docs/index.md).
- **Tutorials:** see [`tutorials/`](tutorials/) for worked, runnable examples —
  a heat equation convergence study, mean-curvature flow of a shrinking sphere,
  a Turing pattern coupled to surface growth, and a distance-function solve —
  each demonstrating the core API and validated against known analytical or
  qualitative results.
- **Citing this software:** see [`CITATION.cff`](CITATION.cff).
