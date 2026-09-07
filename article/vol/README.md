# `article/vol/` — segmented dendritic-spine meshes

Real, segmented dendritic-spine geometries used by
[`../Application_3_realistic_spine.py`](../Application_3_realistic_spine.py)
and the realistic-mesh supplementary tests in
[`../Supplementary_tests/`](../Supplementary_tests/), at three mesh
resolutions:

| Folder | Resolution |
|---|---|
| `spine_sliced_coarse/` | coarse |
| `spine_sliced_intermed/` | intermediate |
| `spine_sliced_fine/` | fine (also referred to as "refined" in filenames) |

Within each resolution folder:

- **`open/`** — the plasma membrane (PM) surface exactly as sliced out of
  the segmentation, with an open cut boundary where it was separated from
  the rest of the dendrite/spine complex. Used as the *surface* compartment
  in the applications (`comp2`/`"surface"`), with the cut edge clamped.
- **`closed/`** — the same PM surface, capped to close the cut into a
  watertight surface. Used by the stationarity/energy-decay supplementary
  tests, which need a boundary-free membrane.
- **`closed/filled/`** — the closed surface filled in with a tetrahedral
  bulk mesh, i.e. an actual volumetric mesh of the spine's cytoplasm. This
  is the `parentmesh` loaded by `Application_3_realistic_spine.py`, since
  that script needs a volume compartment (for the actin reaction-diffusion
  system) in addition to the membrane surface.

## File types

- **`.vol`** — Netgen native mesh format, the format Cosmos/NGSolve loads
  directly (`Mesh("...vol")`).
- **`.h5` / `.xdmf`** — paired HDF5/XDMF files for visualizing the raw
  segmented mesh in ParaView before/independently of the Netgen pipeline.
- **`.vtk`** — VTK export of an intermediate fixing step, for visual
  quality-control inspection.

## The `fixing*.py` scripts

Raw segmented meshes are not guaranteed to have consistent triangle winding
or correct boundary-region tagging, both of which NGSolve/Cosmos rely on
(e.g. a consistent outward normal, or matching boundary names between a
surface mesh and its filled volumetric counterpart). Each folder's
`fixing.py` (or `fixingNEW.py`, an alternate/updated version of the same
step) repairs its raw `*.vol` file using
[`cosmos.utils.mesh_fixing.CosmosAliasMesh`](../../docs/utils/mesh_fixing.md)
and writes out the corresponding `*_fixed.vol` that the application scripts
actually load. Concretely:

- **`open/` and `closed/` fixing.py**: `reorient_surface_triangles_consistently()`
  — fixes inconsistent triangle winding on the surface mesh.
- **`closed/filled/` fixing_tetr.py**: `build_surface_from_volume()` (derive
  the boundary surface from the tetrahedral mesh) +
  `reorient_surface_triangles_consistently(flip=True)`, then
  `mark_cd_elements(...)` against the corresponding `open/` surface mesh so
  the filled volume's boundary tags line up with the surface-only mesh used
  elsewhere.

Only the resulting `*_fixed.vol` files are loaded by the application/test
scripts; the `fixing*.py` scripts themselves are provenance/reproducibility
records of how they were produced, not something you need to re-run unless
regenerating the meshes from new raw segmentation data.
