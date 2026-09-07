"""The contract every concrete ``cosmos.pde`` model must satisfy.

``BasePDEModel`` is what makes ``CosmosStepManager`` able to drive an
arbitrary mix of PDE models without knowing anything about their physics:
``solve_step`` only ever calls ``Initialize``/``PreProcess``/``Solve``/
``PostProcess`` on each one, in that order, so any new model dropped into
``cosmos.pde`` slots into the same loop automatically. Two conventions the
concrete subclasses (``cosmos.pde.adr``, ``cosmos.pde.distance``,
``cosmos.pde.geom_flow``) all follow, but that live in the subclasses
rather than being enforced here:

- ``is_bnd``/``is_vol`` are declared as *class* attributes (not set in
  ``__init__``), since whether a given model type lives on a surface or a
  volume never varies between instances -- ``CosmosModel.create_pde``
  checks these against the target compartment before assembling anything.
- Symbolic forms (``BilinearForm``/``LinearForm``) are built once in
  ``Initialize()`` and only re-assembled (not rebuilt) in ``Solve()``; the
  cost of rebuilding them every step is the reason to keep this pattern
  when adding a new model.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

from cosmos.config.parameters import get_config
from cosmos.core.field import Field

if TYPE_CHECKING:
    from cosmos.core.compartment import CosmosCompartment
    from cosmos.core.model import CosmosModel


class BasePDEModel(ABC):
    """Abstract base class for all PDE models in Cosmos.

    Defines the four-phase lifecycle every concrete model must implement:
    :meth:`Initialize` — set up FE spaces and assemble time-independent forms;
    :meth:`PreProcess` — snapshot the solution before the current solve;
    :meth:`Solve` — assemble and invert the linear system;
    :meth:`PostProcess` — project results for output and apply post-corrections
    (e.g. mass or bound preservation).
    """

    def __init__(
        self,
        name: str = "BasePDEModel",
        model: Optional[CosmosModel] = None,
        compartment: Optional[CosmosCompartment] = None,
    ) -> None:
        self.name = name
        self.model = model
        self.compartment = compartment
        self.params: Dict[str, Any] = {"printing": False}
        self.cfg = get_config()

    @abstractmethod
    def Initialize(self) -> None:
        raise NotImplementedError("Base class Initialize is being called")

    @abstractmethod
    def PreProcess(self) -> None:
        raise NotImplementedError("Base class PreProcess is being called")

    @abstractmethod
    def Solve(self) -> None:
        raise NotImplementedError("Base class Solve is being called")

    @abstractmethod
    def PostProcess(self) -> None:
        raise NotImplementedError("Base class PostProcess is being called")

    def set_params(self, **kwargs: Any) -> None:
        # A concrete model pre-populates self.params in its own __init__
        # with a mix of plain values (fes_order, Dir_bnd, ...) and Field-
        # wrapped ones (diffusion coefficients, boundary data, ...); this
        # dispatch is what makes set_params(d_1=...) update the existing
        # Field in place (so any form already built against it stays valid)
        # while set_params(fes_order=...) just replaces the plain value.
        # Only keys the model itself already declared can be set -- this is
        # deliberately not a place to introduce new parameters.
        for key, value in kwargs.items():
            if key in self.params.keys():
                if isinstance(self.params[key], Field):
                    self.params[key].cf = value
                else:
                    self.params[key] = value
            else:
                raise ValueError(f"Parameter {key} not present in PDEModel")

    def adaptive_timestep_cap(self) -> bool:
        return False

    def reset(self) -> None:
        pass
