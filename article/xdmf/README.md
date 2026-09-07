# `article/xdmf/`

Paired HDF5/XDMF files for visualizing the segmented spine geometry (and a
couple of segmented organelle surfaces) directly in ParaView, independently
of the Netgen `.vol` mesh-fixing pipeline in [`../vol/`](../vol/).

- **`spine_{coarse,coarseNEW,intermed,fine/refined}_*`** — plasma-membrane
  (`PM`) and endoplasmic-reticulum (`ER`) surfaces at each of the mesh
  resolutions used in `../vol/`, both open (`_sliced_PM`) and capped
  (`_sliced_PM_closed`) variants, plus `.vtk` snapshots of a couple of the
  mesh-fixing steps' cut geometry (`*_cut.vtk`).
- **`mito_IM.*` / `mito_OM.*`** — segmented mitochondrial inner/outer
  membrane surfaces. These are not referenced by any of the `Application_*`
  or `Supplementary_tests` scripts in this repository; they appear to be
  additional segmented organelle geometry from the same dataset, kept here
  for completeness/future use rather than something the current simulations
  read from.

None of these files are loaded by the simulation scripts directly -- they
are visualization companions to the `.vol` meshes in `../vol/`.
