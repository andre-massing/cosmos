# read version from installed package
from importlib.metadata import version
__version__ = version("cosmos")

from cosmos.pde.adr.volume.adr_volume_bdf1_model import ADRVolumeBDF1Model
from cosmos.pde.adr.boundary.adr_boundary_bdf1_model import ADRBoundaryBDF1Model
from cosmos.pde.cahn_hilliard.volume.cahn_hilliard_volume_bdf1_model import CahnHilliardVolumeBDF1Model
from cosmos.pde.cahn_hilliard.boundary.cahn_hilliard_boundary_bdf1_model import CahnHilliardBoundaryBDF1Model
from cosmos.pde.mean_curvature.mean_curvature_boundary_bdf1_model import MeanCurvatureBoundaryBDF1Model
from cosmos.pde.mean_curvature.mean_curvature_boundary_stab_bdf1_model import MeanCurvatureBoundaryStabBDF1Model
from cosmos.pde.willmore.willmore_boundary_bdf1_model import WillmoreBoundaryBDF1Model
from cosmos.pde.willmore.willmore_boundary_stab_bdf1_model import WillmoreBoundaryStabBDF1Model
from cosmos.pde.willmore.willmore_boundary_v1_bdf1_model import WillmoreBoundaryV1BDF1Model
from cosmos.pde.willmore.willmore_boundary_v2_bdf1_model import WillmoreBoundaryV2BDF1Model
from cosmos.coupling.deformation_boundary_bdf1_coupling import DeformationBoundaryBDF1Coupling
from cosmos.coupling.displacement_boundary_bdf1_coupling import DisplacementBoundaryBDF1Coupling
from cosmos.coupling.deformation_volume_bdf1_coupling import DeformationVolumeBDF1Coupling
from cosmos.coupling.displacement_volume_bdf1_coupling import DisplacementVolumeBDF1Coupling
from cosmos.io.logger import *
from cosmos.core.mesh import SolverMesh
from cosmos.core.time import SolverTime
from cosmos.core.solver import Solver
from cosmos.config.parameters import Config, get_config, set_config

__all__ = [
    "ADRVolumeBDF1Model",
    "ADRBoundaryBDF1Model",
    "CahnHilliardVolumeBDF1Model",
    "CahnHilliardBoundaryBDF1Model",
    "MeanCurvatureBoundaryBDF1Model",
    "MeanCurvatureBoundaryStabBDF1Model",
    "WillmoreBoundaryBDF1Model",
    "WillmoreBoundaryStabBDF1Model",
    "WillmoreBoundaryV1BDF1Model",
    "WillmoreBoundaryV2BDF1Model",
    "DeformationBoundaryBDF1Coupling",
    "DisplacementBoundaryBDF1Coupling",
    "DeformationVolumeBDF1Coupling",
    "DisplacementVolumeBDF1Coupling",
    "SolverMesh",
    "SolverTime",
    "Solver",
    "Config",
    "get_config",
    "set_config"
]