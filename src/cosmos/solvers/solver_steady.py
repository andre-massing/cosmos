from ngsolve import *
from cosmos.solvers.solver_base import SolverData, BaseSolver
from cosmos.pdes.pde_base import BasePDE
from cosmos.solvers.schemes import Scheme
from cosmos.utils.tools import params_check
from cosmos.solvers.solver_tools import print_mesh_info

class SteadySolver(BaseSolver):

    def __init__(self, mesh, **kwargs):

        accepted_keys = ['verbose']
        defaults = [0]
        params_check(params = kwargs, 
                    accepted_keys = accepted_keys, 
                    defaults = defaults)
        
        self.verbose = kwargs['verbose']
        
        self.data = SolverData(mesh)
        if self.verbose > 0:
            print_mesh_info(self.data.mesh)
        
        self.PDEs = []

    def AddPDE(self, pde, scheme):

        if not isinstance(pde, BasePDE):
            raise TypeError("The added PDE must inherit from BasePDE")
        if not isinstance(scheme, Scheme):
            raise TypeError("The added scheme must inherit from Scheme")
        pde.Initialize(self.data)
        self.PDEs.append([pde, scheme])

        if self.verbose > 0:
            pde.print_info()

    def Solve(self):

        with TaskManager():

            self.PreProcess()
            self.SolveStep()
            self.PostProcess()

    def SolveStep(self):

        for pde, scheme in self.PDEs:
            pde.PreProcess(self.data)
            scheme.Solve(self.data, pde)
            pde.PostProcess(self.data)
                    
    def PreProcess(self):

        pass

    def PostProcess(self):

        for pde, _ in self.PDEs:
            pde.Update(self.data)