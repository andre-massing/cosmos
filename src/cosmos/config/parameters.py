import logging
logger = logging.getLogger(__name__)

from dataclasses import dataclass, field
from typing import Optional
from ngsolve import *
# from myngspy import *

@dataclass
class Config:
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
