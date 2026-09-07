"""Process-wide configuration, read by every ``cosmos.pde`` model.

``cosmos.config`` is deliberately the one subpackage with no dependency on
``cosmos.core`` or ``cosmos.pde`` -- everything else depends on this, not
the other way around. In practice only ``h`` is actually consumed
(``BasePDEModel.__init__`` calls ``get_config()`` and stores it as
``self.cfg``, and the two ``cosmos.pde.adr`` boundary models reference
``self.cfg.h`` directly as a stabilisation length scale); ``seed``/
``buffer``/``precision`` are recorded here but not currently read back
anywhere in ``cosmos``. There is exactly
one ``Config`` instance for the whole process (module-level
``_current_config``), so ``set_config`` affects every model that reads it,
including ones created before the call.
"""

import logging

logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import Optional
from ngsolve import *


@dataclass
class Config:
    """Global configuration dataclass for the Cosmos solver.

    Stores shared numerical settings (mesh-size reference ``h``, floating-point
    precision, buffer size, and an optional random seed) that are read by PDE
    models via :func:`get_config`.
    """

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
            raise ValueError(f"Unknown configuration parameter: {key}")
        setattr(_current_config, key, value)
