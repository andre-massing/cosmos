import logging
logger = logging.getLogger(__name__)

import os
import time
from ngsolve import *
from cosmos.core.mesh import SolverMesh
from cosmos.core.time import SolverTime
from cosmos.core.iterator import SolverIterator
from cosmos.pde.base import BasePDEModel
from typing import Dict

class Solver:

    def __init__(self, solvermesh:SolverMesh, solvertime:SolverTime = None, name:str = 'solver', iter:bool = False):

        if not isinstance(name, str):
            raise Exception('Name must be a string')
        else:
            self.name = name
        
        if not isinstance(solvermesh, SolverMesh):
            raise Exception("The mesh must be a SolverMesh object")
        else:
            self.mesh = solvermesh
            self.ngsmesh = solvermesh.mesh
        if solvertime != None:
            if not isinstance(solvertime, SolverTime):
                raise Exception("The time must be a SolverTime object")
            else:
                self.time = solvertime
        else:
            self.time = SolverTime(dt = 0.0, initial_t=0, final_t=0)

        self.pdes: Dict[str, BasePDEModel] = {}
        self.iterator = SolverIterator(self.pdes, iter)

    def __call__(self):
        return self._generator()

    def _attach_model(self, pde:BasePDEModel, order:int):

        if not isinstance(pde, BasePDEModel):
            raise Exception('The attached model must derive from BasePDEModel')
        elif not isinstance(order, int):
            raise Exception('The order given for the PDE Model has to be an integer')
        else:
            if pde.name in self.pdes.keys():
                raise Exception('PDE Models'' names must be unique in order to avoid conflicts')
            else:
                self.pdes[pde.name] = pde
    
    def _generator(self):

        start_time = time.time()

        with TaskManager():
            while self.time.t.Get()<self.time.final_t:
                self.time.preprocess()
                self.iterator.preprocess()
                self.iterator.solve_step()

                elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                logging.info(
                    f"t = {self.time.t.Get():.2e} | Δt = {self.time.dt.Get():.2e} | Iter {self.time.iter}"
                    f" | Elapsed = {elapsed} \r"
                )

                yield
                self.iterator.postprocess()
                self.time.postprocess()

    def run(self):
        for step in self(): 
                pass

    def save_model_solution(self, pde:str, output_field:str, subdivision = 0, sample_rate = 1):

        if pde in self.pdes.keys():
            if output_field in self.pdes[pde].output_fields.keys():

                field = self.pdes[pde].output_fields[output_field]
                field.save = True
                folder = './' + self.name + '/' + field.name
                os.makedirs(folder, exist_ok = True)
                filename = folder + '/' + field.name + '_vtk'
                field.vtk = VTKOutput(self.ngsmesh, coefs=[field._coef], names =[field.name],
                                      filename= filename, subdivision = subdivision)
                field.sample_rate = sample_rate
            else:
                raise Exception('Output field ' + output_field + ' to be saved is not in PDE model ' + pde)
        else:
            raise Exception('PDE' + pde + ' is not registered as a Model in Solver ' + self.name)

    @property
    def current_time(self):
        return self.time.t.Get()