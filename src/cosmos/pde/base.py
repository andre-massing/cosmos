import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from typing import Dict
from cosmos.core.field import InputField, OutputField
from cosmos.config.parameters import get_config

class BasePDEModel(ABC):

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
        logger.error('Base class PreProcess is being called')

    @abstractmethod
    def Solve(self):
        logger.error('Base class Solve is being called')

    @abstractmethod
    def PostProcess(self):
        logger.error('Base class PostProcess is being called')

    def set_input_fields(self, input_fields: Dict):
        for key, value in input_fields.items():
            if key in self.input_fields.keys():
                self.input_fields[key].cf = value
            else:
                raise Exception('No such field in Model ' + self.name, key)

    def set_input_params(self, input_params: Dict):
        for key, value in input_params.items():
            if key in self.input_params.keys():
                self.input_params[key] = value
            else:
                raise Exception('No such parameter in Model  ' + self.name + ': ' + key)

    def update_input_fields(self):
        for value in self.input_fields.values():
            value.update()

    def get_output_fields(self):
        return self.output_fields