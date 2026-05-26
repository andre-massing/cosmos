from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING
from cosmos.core.field import Field
from cosmos.config.parameters import get_config

if TYPE_CHECKING:
    from cosmos.core.model import CosmosModel
    from cosmos.core.compartment import CosmosCompartment

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
        name: str = 'BasePDEModel',
        model: Optional[CosmosModel] = None,
        compartment: Optional[CosmosCompartment] = None,
    ) -> None:
        self.name = name
        self.model = model
        self.compartment = compartment
        self.params: Dict[str, Any] = {'printing': False}
        self.cfg = get_config()
        self.vtk = None
        self.is_bnd: Optional[bool] = None
        self.is_vol: Optional[bool] = None

    @abstractmethod
    def Initialize(self) -> None:
        raise NotImplementedError('Base class Initialize is being called')

    @abstractmethod
    def PreProcess(self) -> None:
        raise NotImplementedError('Base class PreProcess is being called')

    @abstractmethod
    def Solve(self) -> None:
        raise NotImplementedError('Base class Solve is being called')

    @abstractmethod
    def PostProcess(self) -> None:
        raise NotImplementedError('Base class PostProcess is being called')

    def set_params(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if key in self.params.keys():
                if isinstance(self.params[key], Field):
                    self.params[key].cf = value
                else:
                    self.params[key] = value
            else:
                raise ValueError(f'Parameter {key} not present in PDEModel')

    def adaptive_timestep_cap(self) -> bool:
        return False

    def reset(self) -> None:
        pass
