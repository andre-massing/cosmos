# Abstract base class defining the contract every concrete PDE model in
# cosmos.pde implements: a PreProcess/Solve/PostProcess lifecycle plus named
# input/output field management. See docs/pde_models.md for the shared
# construction pattern followed by all subclasses.

import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from typing import Dict
from cosmos.core.field import InputField, OutputField
from cosmos.config.parameters import get_config

class BasePDEModel(ABC):
    """Abstract base class for all finite-element PDE models attached to a Solver.

    Subclasses are expected to build their own finite element space(s), a solution
    GridFunction (`gfu`), and populate `input_fields`/`output_fields` in their own
    `__init__` (this base constructor only initializes shared bookkeeping); they
    must implement `PreProcess`, `Solve`, and `PostProcess`, called once per time
    step, in that order, by `cosmos.core.iterator.SolverIterator`.
    """

    def __init__(self):
        self.name = 'base_model'
        self.input_params = {}
        self.input_fields: Dict[str, InputField] = {}
        self.output_fields: Dict[str, OutputField] = {}
        self.cfg = get_config()
        self.save = False
        self.vtk = None
        self.VorB = None

    @abstractmethod
    def PreProcess(self):
        """Called once per time step, before Solve(); subclasses typically snapshot
        the previous solution here (`gfu_old.vec.data = gfu.vec.data`)."""
        logger.error('Base class PreProcess is being called')

    @abstractmethod
    def Solve(self):
        """Called once (or repeatedly, under sub-iteration) per time step; subclasses
        refresh input fields, (re-)assemble their forms, and solve for `gfu`."""
        logger.error('Base class Solve is being called')

    @abstractmethod
    def PostProcess(self):
        """Called once per time step, after Solve(); used by models that need to
        advance auxiliary state (e.g. ALEModel commits the new mesh deformation)."""
        logger.error('Base class PostProcess is being called')

    def set_input_fields(self, input_fields: Dict):
        """Overwrite the coefficient of one or more already-registered InputFields
        by name. Raises if a key doesn't match a registered input field."""
        for key, value in input_fields.items():
            if key in self.input_fields.keys():
                self.input_fields[key].cf = value
            else:
                raise Exception('No such field in Model ' + self.name, key)

    def set_input_params(self, input_params: Dict):
        """Overwrite one or more already-registered scalar input parameters by
        name. Raises if a key doesn't match a registered parameter."""
        for key, value in input_params.items():
            if key in self.input_params.keys():
                self.input_params[key] = value
            else:
                raise Exception('No such parameter in Model  ' + self.name + ': ' + key)

    def update_input_fields(self):
        """Refresh every registered InputField's backing GridFunction from its
        current coefficient; typically called at the start of Solve()."""
        for value in self.input_fields.values():
            value.update()

    def get_output_fields(self):
        """Return the dict of named OutputFields exposed by this model."""
        return self.output_fields