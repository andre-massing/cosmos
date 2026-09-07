# `spine_sliced_intermed/closed/filled/`

Intermediate-resolution spine, closed membrane filled in with a tetrahedral
bulk mesh. `spine_intermed_cut_fixed.vol` here is the `parentmesh` loaded
by
[`../../../Application_3_realistic_spine.py`](../../../Application_3_realistic_spine.py).
`fixing_tetr.py` reproduces it from the raw `spine_intermed_cut.vol`,
cross-referencing [`../../open/`](../../open/) so the filled volume's
boundary tags match the open surface mesh. See
[`../../../README.md`](../../../README.md) for the full pipeline
explanation.
