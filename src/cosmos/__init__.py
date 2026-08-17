# Public API of the cosmos package: re-exports the core simulation building blocks
# (Solver, SolverMesh, SolverTime, Config) and every concrete PDE model, so user
# code can simply do `from cosmos import ...`. See docs/README.md for an overview
# and docs/architecture.md for how these pieces fit together.

# read version from installed package
from importlib.metadata import version
__version__ = version("cosmos")

# --- PDE models: advection-diffusion-reaction (volume and boundary variants) ---
from cosmos.pde.adr.volume.adr_volume_bdf1_model import ADRVolumeBDF1Model
from cosmos.pde.adr.volume.adr_volume_bdf2_model import ADRVolumeBDF2Model
from cosmos.pde.adr.volume.adr_volume_stab_bdf1_model import ADRVolumeStabBDF1Model
from cosmos.pde.adr.volume.adr_volume_stab_bdf2_model import ADRVolumeStabBDF2Model
from cosmos.pde.adr.boundary.adr_boundary_bdf1_model import ADRBoundaryBDF1Model
from cosmos.pde.adr.boundary.adr_boundary_bdf2_model import ADRBoundaryBDF2Model
from cosmos.pde.adr.boundary.adr_boundary_stab_bdf1_model import ADRBoundaryStabBDF1Model
from cosmos.pde.adr.boundary.adr_boundary_stab_bdf2_model import ADRBoundaryStabBDF2Model
# --- PDE models: Cahn-Hilliard phase separation (volume and boundary variants) ---
from cosmos.pde.cahn_hilliard.volume.cahn_hilliard_volume_aland_bdf1_model import CahnHilliardVolumeAlandBDF1Model
# from cosmos.pde.cahn_hilliard.boundary.cahn_hilliard_boundary_pol_bdf1_model import CahnHilliardBoundaryPolBDF1Model
from cosmos.pde.cahn_hilliard.boundary.cahn_hilliard_boundary_bachini_bdf1_model import CahnHilliardBoundaryBachiniBDF1Model
from cosmos.pde.cahn_hilliard.boundary.cahn_hilliard_boundary_bachini_log_bdf1_model import CahnHilliardBoundaryBachiniLogBDF1Model
# from cosmos.pde.cahn_hilliard.boundary.cahn_hilliard_boundary_elliott_bdf1_model import CahnHilliardBoundaryElliottBDF1Model
# --- PDE models: mean curvature flow (boundary only) ---
from cosmos.pde.mean_curvature.mean_curvature_boundary_bdf1_model import MeanCurvatureBoundaryBDF1Model
# --- PDE models: Willmore / Helfrich bending flow (boundary only) ---
from cosmos.pde.willmore.willmore_boundary_bdf1_model import WillmoreBoundaryBDF1Model
from cosmos.pde.willmore.willmore_boundary_ap_bdf1_model import WillmoreBoundaryAPBDF1Model
from cosmos.pde.willmore.willmore_boundary_vp_bdf1_model import WillmoreBoundaryVPBDF1Model
from cosmos.pde.willmore.willmore_boundary_apvp_bdf1_model import WillmoreBoundaryAPVPBDF1Model
from cosmos.pde.willmore.willmore_boundary_inex_bdf1_model import WillmoreBoundaryInexBDF1Model

# --- Logging setup, mesh-motion coupling, and core simulation building blocks ---
from cosmos.io.logger import *
from cosmos.coupling.ale_model import ALEModel
from cosmos.core.mesh import SolverMesh
from cosmos.core.time import SolverTime
from cosmos.core.solver import Solver
from cosmos.config.parameters import Config, get_config, set_config

# Names exported by `from cosmos import *`.
__all__ = [
    "ADRVolumeBDF1Model",
    "ADRVolumeBDF2Model",
    "ADRVolumeStabBDF1Model",
    "ADRVolumeStabBDF2Model",
    "ADRBoundaryBDF1Model",
    "ADRBoundaryBDF2Model",
    "ADRBoundaryStabBDF1Model",
    "ADRBoundaryStabBDF2Model",
    "CahnHilliardVolumeAlandBDF1Model",
    "CahnHilliardBoundaryBachiniBDF1Model",
    "CahnHilliardBoundaryBachiniLogBDF1Model",
    # "CahnHilliardBoundaryPolBDF1Model",
    # "CahnHilliardBoundaryElliottBDF1Model",
    "MeanCurvatureBoundaryBDF1Model",
    "WillmoreBoundaryBDF1Model",
    "WillmoreBoundaryAPBDF1Model",
    "WillmoreBoundaryVPBDF1Model",
    "WillmoreBoundaryAPVPBDF1Model",
    "WillmoreBoundaryInexBDF1Model",
    "ALEModel",
    "SolverMesh",
    "SolverTime",
    "Solver",
    "Config",
    "get_config",
    "set_config"
]