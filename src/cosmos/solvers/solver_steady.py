from ngsolve import *
from cosmos.solvers.solver_base import SolData
from cosmos.pdes.pde_base import BasePDE
from cosmos.solvers.schemes import Scheme
from cosmos.utils.tools import params_check
from cosmos.solvers.solver_tools import print_mesh_info

class SteadySolver():

    def __init__(self, mesh, scheme, **kwargs):

        if not isinstance(scheme, Scheme):
            raise TypeError("The added scheme must inherit from Scheme")
        self.scheme = scheme

        accepted_keys = ['verbose']
        defaults = [0, False]
        params_check(params = kwargs, 
                    accepted_keys = accepted_keys, 
                    defaults = defaults)
        
        self.verbose = kwargs['verbose']
        
        self.data = SolData(mesh)
        if self.verbose > 0:
            print_mesh_info(self.data.mesh)
        
        self.PDEs = []

    def AddPDE(self, *args):

        for pde in args:

            if not isinstance(pde, BasePDE):
                raise TypeError("The added PDE must inherit from BasePDE")
            self.PDEs.append(pde)

            if self.verbose > 0:
                pde.print_info()

    def Solve(self):

        with TaskManager():

            self.PreProcess()

            for pde in self.PDEs:

                pde.Initialize(self.data)

                self.scheme.Solve(self.data, pde)

            self.PostProcess()
                    
    def PreProcess(self):

        pass

    def PostProcess(self):

        for pde in self.PDEs:

            pde.Update(self.data)