from ngsolve import *
from cosmos.solvers.solver_base import SolverData, SolverALE
from cosmos.pdes.pde_base import BasePDE
from cosmos.solvers.schemes import Scheme
from cosmos.utils.tools import params_check
from cosmos.solvers.solver_steady import SteadySolver
from cosmos.solvers.solver_tools import print_mesh_info
from tqdm import tqdm

class UnsteadySolver(SteadySolver):

    def __init__(self, mesh, dt, t, T, **kwargs):

        accepted_keys = ['verbose']
        defaults = [0]
        params_check(params = kwargs, 
                    accepted_keys = accepted_keys, 
                    defaults = defaults)
        
        self.verbose = kwargs['verbose']
        
        self.data = SolverData(mesh, dt=dt, t=t, T=T)
        self.t0 = self.data.t.Get()
        if self.verbose > 0:
            print_mesh_info(self.data.mesh)
        self.ale = SolverALE(mesh)
        
        self.PDEs = []

    def SolveStep(self):

        for pde, scheme in self.PDEs:
            scheme.Solve(self.data, pde, self.ale)
            self.ale.UpdateALE()

    def __generator__(self):

        with TaskManager():

            max_steps = int(self.data.T/self.data.dt.Get())

            if self.verbose>0:
                print('Initializing...')

            self.Initialize()

            if self.verbose>0:
                print('Initialization successful \n\
                        N.', len(self.PDEs), ' PDEs have been initialized correctly')

            yield

            if self.verbose>0:
                print('-'*10, '\nStarting the simulation...')

            for step in tqdm(range(max_steps), desc="\t Running Simulation...", 
                ascii=False, ncols=75):

                self.data.iter += 1

                self.PreProcess()
                self.SolveStep()

                self.data.t.Set(self.data.t.Get() + self.data.dt.Get())

                self.PostProcess()

                yield

            if self.verbose>0:
                print('Simulation concluded successfully')
                print('-'*10)

    def __call__(self):
        return self.__generator__()
    
    def PreProcess(self):

        self.data.prev_dt.append(self.data.dt.Get())    
        if len(self.data.prev_dt)>6:
            self.data.prev_dt.pop(0)
        self.ale.UpdateALE()

    def Initialize(self):

        self.ale.prev_d.append(self.ale.deformation.vec.Copy())
        self.ale.prev_v.append(self.ale.velocity.vec.Copy())
        for pde, _ in self.PDEs:
            pde.Update(self.data, self.ale)

    def Solve(self):

        self.data.t.Set(self.t0)
        for step in self(): 
                pass
        
    def PostProcess(self):

        self.ale.prev_d.append(self.ale.deformation.vec.Copy())
        self.ale.prev_v.append(self.ale.velocity.vec.Copy())    
        if len(self.ale.prev_d)>6:
            self.ale.prev_d.pop(0)
            self.ale.prev_v.pop(0)
        for pde, _ in self.PDEs:
            pde.Update(self.data, self.ale)