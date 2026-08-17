# Process-wide simulation configuration: mesh-size symbol, time-level buffer size,
# random seed and numerical precision, shared by all core objects and PDE models.

import logging
logger = logging.getLogger(__name__)

from dataclasses import dataclass, field
from typing import Optional
from ngsolve import *
# from myngspy import *

@dataclass
class Config:
    """Container for global simulation settings.

    Attributes:
        h: NGSolve mesh-size symbol used in stabilization/penalty terms.
        seed: Optional random seed (reserved for reproducible randomness).
        buffer: Number of previous time levels retained by SolverMesh/SolverTime
            (must be >= the number of previous levels a time-discretization needs,
            e.g. 2 for BDF2).
        precision: Reserved floating-point precision setting.
    """
    # h = MyMeshSize()
    h = specialcf.mesh_size
    seed: Optional[int] = None
    buffer: int = 2
    precision: str = "float64"

# single, shared config object
_current_config = Config()

def get_config() -> Config:
    """Return the current config (read-only use)."""
    return _current_config

def set_config(**kwargs):
    """Update config parameters."""
    for key, value in kwargs.items():
        if not hasattr(_current_config, key):
            logger.error("Unknown config parameter: {key}")
        setattr(_current_config, key, value)
