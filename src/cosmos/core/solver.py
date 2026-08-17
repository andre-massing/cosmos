# Top-level simulation orchestrator: owns the mesh, the clock, and the registry of
# attached PDE models, and drives the time-stepping loop.

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
import traceback

class Solver:
    """Orchestrates a simulation: owns a SolverMesh and SolverTime, holds the
    registry of attached PDE models, and exposes the time-stepping loop as a
    generator (`solver()`) or a blocking call (`solver.run()`)."""

    def __init__(self, solvermesh:SolverMesh, solvertime:SolverTime = None, name:str = 'solver',
                 iter:bool = False, printing:bool = False):
        """
        Args:
            solvermesh: The mesh (and deformation state) the simulation runs on.
            solvertime: The time-stepping manager; defaults to a zero-length,
                single-instant SolverTime if omitted.
            name: Identifies this solver, used to namespace output folders/files.
            iter: If True, PDE models are solved with fixed-point sub-iteration each
                step (see SolverIterator) to resolve implicit multi-physics coupling.
            printing: If True, print progress information during the run.
        """

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
        """Return the step-by-step simulation generator (see `_generator`)."""
        return self._generator()

    def _attach_model(self, pde:BasePDEModel, order:int):
        """Register a PDE model with this solver under a given execution order.

        Called by PDE model constructors (not typically by user code directly).
        Raises if `pde` is not a BasePDEModel, `order` is not an int, or a model
        with the same name is already registered.
        """

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
        """Generator driving the simulation one time step per iteration.

        Each iteration runs SolverTime/SolverIterator pre/postprocess and
        Solver-Iterator's solve_step (see cosmos.core.iterator.SolverIterator),
        writes periodic output, then yields control back to the caller. On
        uncaught exception, the last state is saved and an error report is written
        to `<output_folder>/<name>/<name>.err` before the exception propagates.
        """

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
                        print(f"t = {self.time.t.Get():.3e} | time-step = {dt:.3e} | Iter {self.time.iter}"
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
                traceback.print_exc(file=fw)

    def run(self):
        """Run the simulation to completion, discarding per-step control (use
        `for _ in solver():` directly instead if intermediate access is needed)."""
        for step in self():
                pass

    def save_model_solution(self, pde: BasePDEModel, subdivision = 0):
        """Opt an attached PDE model into periodic VTK output of its output_fields.

        Must be called after `output_params()` has set `self.output_folder`. Raises
        if `pde` is not registered with this solver.
        """

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
        """Write a VTK output frame for every model with `save == True`, every
        `output_sample_rate` iterations."""
        for _, pde in self.pdes.items():
            if pde.save:
                if self.time.iter % self.output_sample_rate == 0:
                    pde.vtk.Do(time = self.time.t.Get(), vb = pde.VorB)

    def output_params(self, folder, sample_rate):
        """Configure the output folder and how often (in iterations) results are
        written; must be called before `save_model_solution()`."""
        self.output_folder = folder
        self.output_sample_rate = sample_rate

    @property
    def current_time(self):
        """The current simulation time."""
        return self.time.t.Get()