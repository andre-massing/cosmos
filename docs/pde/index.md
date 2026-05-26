# cosmos.pde

The `pde` subpackage contains the concrete PDE model classes that plug into a
`CosmosModel`.  All concrete models inherit from
[`BasePDEModel`](base.md) and share the same four-phase lifecycle.

---

## Inheritance diagram

```
BasePDEModel (ABC)
├── ADRBoundarySystemBDF1Model   (adr/boundary — no stabilisation)
├── ADRBoundarySystemBDF1Model   (adr/boundary — gradient-jump stabilisation)
├── ADRVolumeSystemBDF1Model     (adr/volume)
├── DistanceVolumeModel          (distance/volume)
├── GeometricalFlowModel         (willmore)
└── GeometricalFlowStationaryModel (willmore)
```

---

## Module index

| File | Class | Domain | Description |
|---|---|---|---|
| [`pde/base`](base.md) | `BasePDEModel` | — | Abstract lifecycle interface |
| [`pde/adr/boundary — nostab`](adr_boundary.md#adrboundarysystembdf1model-no-stabilisation) | `ADRBoundarySystemBDF1Model` | surface | ADR system, standard SIPG |
| [`pde/adr/boundary — stab`](adr_boundary.md#adrboundarysystembdf1model-gradient-jump-stabilisation) | `ADRBoundarySystemBDF1Model` | surface | ADR system, gradient-jump stabilisation |
| [`pde/adr/volume`](adr_volume.md) | `ADRVolumeSystemBDF1Model` | volume | ADR system, SIP, volume domain |
| [`pde/distance/volume`](distance.md) | `DistanceVolumeModel` | volume | Smoothed distance function |
| [`pde/willmore`](willmore.md#geometricalflowmodel) | `GeometricalFlowModel` | surface | Willmore / mean-curvature flow |
| [`pde/willmore`](willmore.md#geometricalflowstationarymodel) | `GeometricalFlowStationaryModel` | surface | Flow + co-evolved spontaneous curvature |

---

## Common parameters

All concrete ADR and flow models expose a `params` dictionary.  Use
`pde.set_params(key=value)` to set parameters *after* construction but
*before* `Initialize()` is called (or before a `run()`).

Entries that hold a [`Field`](../core/field.md) accept any value that `Field`
wraps (number, `CoefficientFunction`, `GridFunction`, or callable).
