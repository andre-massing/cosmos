# cosmos.pde

The `pde` subpackage contains the concrete PDE model classes that plug into a
`CosmosModel`.  All concrete models inherit from
[`BasePDEModel`](base.md) and share the same four-phase lifecycle.

---

## Inheritance diagram

```
BasePDEModel (ABC)
├── ADRBoundarySystemBDF1Model       (adr — no stabilisation)
├── ADRBoundarySystemBDF1StabModel   (adr — gradient-jump stabilisation)
├── ADRVolumeSystemBDF1Model         (adr)
├── DistanceVolumeModel              (distance)
├── GeometricalFlowModel             (geom_flow)
└── GeometricalFlowStationaryModel   (geom_flow)
```

---

## Module index

| File | Class | Domain | Description |
|---|---|---|---|
| [`pde/base`](base.md) | `BasePDEModel` | — | Abstract lifecycle interface |
| [`pde/adr — nostab`](adr_boundary.md#adrboundarysystembdf1model-no-stabilisation) | `ADRBoundarySystemBDF1Model` | surface | ADR system, standard SIPG |
| [`pde/adr — stab`](adr_boundary.md#adrboundarysystembdf1stabmodel-gradient-jump-stabilisation) | `ADRBoundarySystemBDF1StabModel` | surface | ADR system, gradient-jump stabilisation |
| [`pde/adr`](adr_volume.md) | `ADRVolumeSystemBDF1Model` | volume | ADR system, SIP, volume domain |
| [`pde/distance`](distance.md) | `DistanceVolumeModel` | volume | Smoothed distance function |
| [`pde/geom_flow`](geom_flow.md#geometricalflowmodel) | `GeometricalFlowModel` | surface | Willmore / mean-curvature flow |
| [`pde/geom_flow`](geom_flow.md#geometricalflowstationarymodel) | `GeometricalFlowStationaryModel` | surface | Flow + co-evolved spontaneous curvature |

All six classes (plus `BasePDEModel`) are also re-exported directly from
`cosmos.pde`, e.g. `from cosmos.pde import ADRVolumeSystemBDF1Model`.

---

## Common parameters

All concrete ADR and flow models expose a `params` dictionary.  Use
`pde.set_params(key=value)` to set parameters *after* construction but
*before* `Initialize()` is called (or before a `run()`).

Entries that hold a [`Field`](../core/field.md) accept any value that `Field`
wraps (number, `CoefficientFunction`, `GridFunction`, or callable).
