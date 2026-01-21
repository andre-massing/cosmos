import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from typing import Dict
from cosmos.core.field import InputField, OutputField, Field
from cosmos.config.parameters import get_config
from cosmos.core.model import CosmosModel
from cosmos.core.compartment import CosmosCompartment

class BasePDEModel(ABC):

    def __init__(self, name = 'BasePDEModel', model:CosmosModel = None, compartment: CosmosCompartment = None):

        self.name = name
        self.model = model
        self.compartment = compartment
        self.params = {'printing': False}
        self.output_fields: Dict[str, OutputField] = {}
        self.cfg = get_config()
        self.vtk = None
        self.is_bnd = None
        self.is_vol = None

    @abstractmethod
    def Initialize(self):
        raise Exception('Base class Initialize is being called')

    @abstractmethod
    def PreProcess(self):
        raise Exception('Base class PreProcess is being called')

    @abstractmethod
    def Solve(self):
        raise Exception('Base class Solve is being called')

    @abstractmethod
    def PostProcess(self):
        raise Exception('Base class PostProcess is being called')

    def set_params(self, **kwargs):
        
        for key, value in kwargs.items():
            if key in self.params.keys():
                if isinstance(self.params[key], Field):
                    self.params[key].cf = value
                else:
                    self.params[key] = value
            else:
                raise Exception(f'Parameter {key} not present in PDEModel')