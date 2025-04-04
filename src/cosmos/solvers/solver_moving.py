from ngsolve import *
from cosmos.solvers.solver_base import SolData
from cosmos.pdes.pde_base import BasePDE
from cosmos.solvers.schemes import Scheme
from cosmos.solvers.coupling_schemes import CouplingScheme
from cosmos.utils.tools import params_check
from cosmos.solvers.solver_unsteady import UnsteadySolver
from cosmos.solvers.solver_tools import print_mesh_info
from tqdm import tqdm
from ngsolve.webgui import Draw

class DisplData():

    def __init__(self, mesh):
        
        self.dX = GridFunction(VectorH1(mesh))
        self.dX_old = GridFunction(VectorH1(mesh))
        self.V = GridFunction(VectorH1(mesh))
        self.X = GridFunction(VectorH1(mesh))

class MovingSolver(UnsteadySolver):

    def __init__(self, mesh, dt, t, T, coupling, **kwargs):

        if not isinstance(coupling, CouplingScheme):
            raise TypeError("The added coupling must inherit from CouplingScheme")
        self.coupling = coupling

        accepted_keys = ['verbose']
        defaults = [0]
        params_check(params = kwargs, 
                    accepted_keys = accepted_keys, 
                    defaults = defaults)
        
        self.verbose = kwargs['verbose']
        
        self.data = SolData(mesh, dt=dt, t=t, T=T)
        self.t0 = self.data.t.Get()
        if self.verbose > 0:
            print_mesh_info(self.data.mesh)
        self.PDEs = []

        self.dX_PDEs = []
        self.dX_data = DisplData(self.data.mesh)

    def AddPDE(self, pde, scheme, dX = False):

        if not isinstance(pde, BasePDE):
            raise TypeError("The added PDE must inherit from BasePDE")
        if not isinstance(scheme, Scheme):
            raise TypeError("The added scheme must inherit from Scheme")
        
        if dX:
            self.dX_PDEs.append([pde, scheme])
        else:
            self.PDEs.append([pde, scheme])

        if self.verbose > 0:
            pde.print_info()

    def SolveStep(self):

        self.coupling.SolveStep(self.dX_data, self.dX_PDEs, self.data, self.PDEs)

    def Initialize(self):

        for pde, _ in self.dX_PDEs:
            pde.Initialize(self.data)
            pde.Update(self.data)

        for pde, _ in self.PDEs:
            pde.Initialize(self.data)
            pde.Update(self.data)
        
    def PostProcess(self):

        for pde, _ in self.dX_PDEs:
            pde.Update(self.data)
        
        for pde, _ in self.PDEs:
            pde.Update(self.data)