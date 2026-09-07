"""Cosmos: finite-element solvers for moving-domain/moving-surface PDEs.

The package is organised by role, not by application:

- ``cosmos.config``   -- process-wide numerical settings (``Config``), shared
  by every PDE model via :func:`cosmos.config.parameters.get_config`.
- ``cosmos.core``     -- the simulation runtime: :class:`~cosmos.core.model.CosmosModel`
  (the top-level object a user constructs) and the managers it owns
  internally (time, ALE mesh motion, per-step solve dispatch, I/O), plus the
  ``CosmosCompartment``/``Field`` building blocks PDE models are built on.
- ``cosmos.pde``      -- concrete PDE models (advection-diffusion-reaction,
  distance functions, geometrical/Willmore flow), all implementing the
  four-phase lifecycle defined by :class:`cosmos.pde.base.BasePDEModel`.
- ``cosmos.utils``    -- geometry generation and mesh repair, independent of
  the simulation runtime above.

A typical script only touches ``CosmosModel`` plus one or more classes from
``cosmos.pde``; the manager classes in ``cosmos.core`` are constructed
automatically by ``CosmosModel.__init__`` and are not meant to be
instantiated directly. See ``docs/index.md`` for the full quick-start.
"""

# read version from installed package
from importlib.metadata import version

__version__ = version("cosmos")

from cosmos.config.parameters import Config, get_config, set_config

__all__ = ["Config", "get_config", "set_config"]
