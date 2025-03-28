from ngsolve import *
from cosmos.solvers.solver_base import SolData
from cosmos.pdes.pde_base import BasePDE
from cosmos.solvers.schemes import Scheme
from cosmos.utils.tools import params_check
from cosmos.solvers.solver_steady import SteadySolver
from cosmos.solvers.solver_tools import print_mesh_info
import tqdm

class UnsteadySolver(SteadySolver):

    def __init__(self, mesh, dt, t, T, scheme, **kwargs):

        if not isinstance(scheme, Scheme):
            raise TypeError("The added scheme must inherit from Scheme")
        self.scheme = scheme

        accepted_keys = ['verbose']
        defaults = [0, False]
        params_check(params = kwargs, 
                    accepted_keys = accepted_keys, 
                    defaults = defaults)
        
        self.verbose = kwargs['verbose']
        
        self.data = SolData(mesh, dt=dt, t=t, T=T)
        if self.verbose > 0:
            print_mesh_info(self.data.mesh)
        
        self.PDEs = []

        self.iter = 0

    def SolveStep(self):

        for pde in self.PDEs:

            self.scheme.Solve(self.data, pde)

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

            if self.verbose>0:

                for step in tqdm(range(max_steps), desc="\t Running Simulation...", 
                    ascii=False, ncols=75):

                    self.iter += 1

                    self.PreProcess()
                    self.SolveStep()

                    self.data.t.Set(self.data.t.Get() + self.data.dt.Get())

                    self.PostProcess()

                    yield

            else:

                for step in range(max_steps):

                    self.iter += 1

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

    def Initialize(self):

        for pde in self.PDEs:

            pde.Initialize(self.data)

    def Solve(self):

        for step in self(): 
                pass
        
    def PostProcess(self):
        
        for pde in self.PDEs:

            pde.Update(self.data)