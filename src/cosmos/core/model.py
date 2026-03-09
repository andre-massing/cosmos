import logging
logger = logging.getLogger(__name__)

import os
import time
from dataclasses import dataclass
from collections import Counter
from ngsolve import *
import traceback

from cosmos.core.time_manager import CosmosTimeManager
from cosmos.core.step_manager import CosmosStepManager
from cosmos.core.ale_manager import CosmosALEManager, CosmosBndALEField, CosmosVolALEField
from cosmos.io.io_manager import CosmosIOManager
from cosmos.core.compartment import CosmosCompartment

from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from cosmos.pde.base import BasePDEModel

class CosmosModel:

    def __init__(self, name: str, parentmesh: Mesh, **kwargs):

        self.name = name
        self.parentmesh = parentmesh
        if kwargs:
            self.params = kwargs
        else:
            self.params = {}

        # mesh-related properties
        self.dim = self.parentmesh.dim
        self.geo_order = self.parentmesh.GetCurveOrder()
        self.num_elem = self.parentmesh.ne
        self.num_faces = self.parentmesh.nface
        self.num_facets = self.parentmesh.nfacet
        self.num_edges = self.parentmesh.nedge
        self.num_vertices = self.parentmesh.nv
        self.is_bnd = self.num_elem == 0
        self.is_vol = self.num_elem != 0
        self.vol_ids = set(Counter(self.parentmesh.GetMaterials()).keys())
        self.bnd_ids = set(Counter(self.parentmesh.GetBoundaries()).keys())
        self.bbnd_ids = set(Counter(self.parentmesh.GetBBoundaries()).keys())

        # model components
        self.compartments: List[CosmosCompartment] = []
        self.pdes:  List[BasePDEModel] = []
        self.pdes_init:  List[BasePDEModel] = []
        self.pdes_pre:  List[BasePDEModel] = []
        self.pdes_post:  List[BasePDEModel] = []
        self.ales = []
        self.time = CosmosTimeManager(self.params)
        self.step = CosmosStepManager(self.params)
        self.ale = CosmosALEManager(self, self.params)
        self.io = CosmosIOManager(self.params)

        self.dt = self.time.dt
        self.t = self.time.t

    def initialize(self):

        self.time.initialize()
        self.ale.initialize(self)
        self.step.initialize(self)
        self.io.initialize(self)

    def __call__(self):
        return self._generator()
    
    def _generator(self):

        with TaskManager():

            self.initialize()
            self.io.save_step_data(self)

            yield
            
            while self.t.Get()<= self.time.t1:

                self.step.solve_step(self)
                self.time.next()
                self.io.save_step_data(self)

                self.ale.finalize(self)

                yield

            self.io.finalize(self)

    def run(self):
        for step in self(): 
                pass
        
    def set_params(self, force:bool = False, **kwargs):
        
        for key, value in kwargs.items():
            if key in self.params.keys() and not force:
                raise Exception(f'Parameter {key} already set in the model, use flag force = True to force the behavior')
            elif key in self.params.keys() and force:
                self.params[key] = value
            else:
                self.params[key] = value
        
    def create_compartment(self, name, **kwargs):

        if any(name==x.name for x in self.compartments):
            raise Exception(f'Unique names must be assigned to compartments, name {name} already used')
        compartment = CosmosCompartment(name = name, model = self, **kwargs)
        self.compartments.append(compartment)
        return compartment
    
    def create_pde(self, name, pde_model: "BasePDEModel", compartment:CosmosCompartment,  ale_type:int , **kwargs):
        pde = pde_model(name, self, compartment, **kwargs)
        if pde.is_bnd and compartment.is_vol:
            raise Exception(f'Boundary PDE {pde_model} is trying to be imposed on a non-boundary domain {compartment.name} or the opposite')
        elif pde.is_vol and compartment.is_bnd:
            raise Exception(f'Volume PDE {pde_model} is trying to be imposed on a boundary domain {compartment.name} or the opposite')
        else:
            compartment.pdes.append(pde)
            if ale_type == -1:
                self.pdes_init.append(pde)
            elif ale_type == 0:
                self.pdes_pre.append(pde)
            elif ale_type == 1:
                self.pdes_post.append(pde)
            else:
                raise Exception('Parameter pde-type must be in the values {-1 ,0, 1}')
            self.pdes.append(pde)
        return pde
    
    def create_ale(self, name:str, compartment:CosmosCompartment, **kwargs):
        if any(x == y for z in self.ales for x in compartment.domain_id.split('|') for y in z.compartment.boundary_id.split('|')):
            raise Exception(f'ALE motion for domain {compartment.domain_id} (or part of it) has already been set')
        else:
            if any(name == x.name for x in self.ales):
                raise Exception(f'Name {name} for ALE motion has already been used. Names must be unique')
            else:
                if compartment.is_bnd:
                    ale = CosmosBndALEField(name, self, compartment)
                else:
                    ale = CosmosVolALEField(name, self, compartment)
                self.ales.append(ale)
                compartment.ale = ale
        return ale
    
    def print_model_data(self):
        self.io.print_model_data(self)

    def print_step_data(self):
        self.io.print_step_data(self)