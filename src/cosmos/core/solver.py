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

    def __init__(self, solvermesh:SolverMesh, solvertime:SolverTime = None, name:str = 'solver',
                 iter:bool = False, printing:bool = False):

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

        self.output_folder = ''
        self.output_sample_rate = None

        self.printing = printing

        if self.printing:
            self.mesh.print_info()
            self.time.print_info()

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

        if self.printing:
            print('\n !!! SIMULATION HAS STARTED !!! \n')
            print('The solver name is: ', self.name)
            print('The solver contains N.', len(self.pdes), ' PDE solvers:')
            for pde_name, pde in self.pdes.items():
                print('\t - Solver ', pde_name, ' with execution order', pde.model_order)
            print('\n')
            if self.output_folder:
                print('Results are saved in: ', self.output_folder)
                print('Results printing sample rate is: ', self.output_sample_rate, '\n')

        start_time = time.time()

        try:
            self._save_pdes_solutions()
            with TaskManager():
                while self.time.t.Get()<self.time.final_t:

                    self.time.preprocess()
                    self.iterator.preprocess()
                    self.iterator.solve_step()
                    self.iterator.postprocess()
                    self.time.postprocess()
                    self._save_pdes_solutions()

                    elapsed_hms = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                    elapsed = time.time() - start_time
                    dt = self.time.dt.Get()

                    if self.printing:
                        print(f"t = {self.time.t.Get():.3e} | Δt = {dt:.3e} | Iter {self.time.iter}"
                            f" | Elapsed = {elapsed_hms}  | Avg. Iter Time = {elapsed/self.time.iter:.3e}", end = '\r')

                    yield
            
            if self.printing:
                print('\n\n !!! SIMULATION CONCLUDED SUCCESSFULLY !!! \n')
        except Exception as e:
            self._save_pdes_solutions()
            print('\n\n !!! SIMULATION CONCLUDED WITH ERROR !!!')
            folder = os.path.join(self.output_folder, self.name)
            os.makedirs(folder, exist_ok = True)
            file = os.path.join(self.output_folder, self.name, self.name +'.err')
            with open(file, 'w') as fw:
                fw.write('Simulation terminated with error\n')
                fw.write('Time: ' +  str(self.time.t.Get()) +', iter: '+ str(self.time.iter) + '\n')
                fw.write('Cause: ' + str(e))

    def run(self):
        for step in self(): 
                pass

    def save_model_solution(self, pde: BasePDEModel, subdivision = 0):

        if pde.name in self.pdes.keys():
            pde.save = True
            folder = os.path.join(self.output_folder, self.name, pde.name)
            os.makedirs(folder, exist_ok = True)
            filename = os.path.join(folder, 'vtk_' + pde.name)
            pde.vtk = VTKOutput(self.ngsmesh,
                                coefs=[out_f.cf for out_f in pde.output_fields.values()],
                                names =[out_f.name for out_f in pde.output_fields.values()],
                                filename= filename, 
                                subdivision = subdivision)
        else:
            raise Exception('PDE ' + pde.name + ' is not registered as a Model in Solver ' + self.name)
        
    def _save_pdes_solutions(self):
        for _, pde in self.pdes.items():
            if pde.save:
                if self.time.iter % self.output_sample_rate == 0:
                    pde.vtk.Do(time = self.time.t.Get(), vb = pde.VorB)

    def output_params(self, folder, sample_rate):
        self.output_folder = folder
        self.output_sample_rate = sample_rate

    @property
    def current_time(self):
        return self.time.t.Get()