"""Every concrete PDE model, flattened to one import level.

The underlying files are nested by family (``cosmos.pde.adr``,
``.distance``, ``.geom_flow``); this module exists purely so a script can
write ``from cosmos.pde import ADRVolumeSystemBDF1Model`` instead of the
full submodule path. All seven names below are ``BasePDEModel`` subclasses
(or ``BasePDEModel`` itself) -- see that module for the shared lifecycle.
"""

from cosmos.pde.base import BasePDEModel

from cosmos.pde.adr.adr_volume_system_bdf1_model import ADRVolumeSystemBDF1Model
from cosmos.pde.adr.adr_boundary_system_bdf1_model_nostab import ADRBoundarySystemBDF1Model
from cosmos.pde.adr.adr_boundary_system_bdf1_model_stab import ADRBoundarySystemBDF1StabModel
from cosmos.pde.distance.distance_volume_model import DistanceVolumeModel
from cosmos.pde.geom_flow.geometrical_flow_model import GeometricalFlowModel
from cosmos.pde.geom_flow.geometrical_flow_stationary_model import GeometricalFlowStationaryModel

__all__ = [
    "BasePDEModel",
    "ADRVolumeSystemBDF1Model",
    "ADRBoundarySystemBDF1Model",
    "ADRBoundarySystemBDF1StabModel",
    "DistanceVolumeModel",
    "GeometricalFlowModel",
    "GeometricalFlowStationaryModel",
]
