from ngsolve import *
from cosmos.pdes.pde_base import BasePDE
from cosmos.solvers.tools import print_mesh_info
from collections import Counter
from tqdm import tqdm
from cosmos.pdes.ale import ale
import time

class BaseSolver():

    def __init__(self, **kwargs):
        raise NotImplementedError

    def AddPDE(self, *args):
        raise NotImplementedError

    def Solve(self):
        raise NotImplementedError
                    
    def PreProcess(self):
        raise NotImplementedError

    def PostProcess(self):
        raise NotImplementedError

class Static(BaseSolver):

    def __init__(self, mesh, verbose = 0):
        
        self.verbose = verbose
        
        self.mesh = mesh
        if self.mesh.ne == 0:
            self.domain_markers = list(Counter(self.mesh.GetBoundaries()).keys())
            self.boundary_markers = list(Counter(self.mesh.GetBBoundaries()).keys())
        else:
            self.domain_markers = list(Counter(self.mesh.GetMaterials()).keys())
            self.boundary_markers = list(Counter(self.mesh.GetBoundaries()).keys())
        if self.verbose > 0:
            print_mesh_info(self.mesh)
        self.ale = ale(self)
        
        self.PDEs = []

    def AddPDE(self, pde):

        if not isinstance(pde, BasePDE):
            raise TypeError("The added PDE must inherit from BasePDE")
        pde.Initialize(self)
        self.PDEs.append(pde)

        if self.verbose > 0:
            pde.print_info()

    def Solve(self):

        with TaskManager():

            self.PreProcess()
            self.SolveStep()
            self.PostProcess()

    def SolveStep(self):
        for pde in self.PDEs:
            pde.Solve(self)
            self.ale.update_ale()
                    
    def PreProcess(self):
        pass

    def PostProcess(self):
        for pde in self.PDEs:
            pde.Update(self)

class Dynamic(Static):

    def __init__(self, mesh, dt, t, T, verbose = 0, postprocess = False):

        super().__init__(mesh, verbose)

        self.t = t
        self.dt = dt
        self.prev_dt = []
        self.T = T
        self.iter = 0
        self.t0 = self.t.Get()
        self.prev_def = []
        self.prev_vel = []
        self.prev_mat_vel = []
        if self.verbose > 0:
            print_mesh_info(self.mesh)
        self.postprocess = postprocess

    def __generator__(self):

        with TaskManager():

            max_steps = int(self.T/self.dt.Get())

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

                self.iter += 1
                self.PreProcess()
                self.SolveStep()
                self.t.Set(self.t.Get() + self.dt.Get())
                self.PostProcess()

                yield

            if self.verbose>0:
                print('Simulation concluded successfully')
                print('-'*10)

    def __call__(self):
        return self.__generator__()
    
    def Initialize(self):

        self.prev_def.append(self.ale.deformation.vec.Copy())
        self.prev_vel.append(self.ale.velocity.vec.Copy())
        self.prev_mat_vel.append(self.ale.material_velocity.vec.Copy())
        for pde in self.PDEs:
            pde.Update(self)
    
    def PreProcess(self):

        self.prev_dt.append(self.dt.Get())    
        if len(self.prev_dt)>6:
            self.prev_dt.pop(0)
        self.ale.deformation_old.vec.data = self.ale.deformation.vec.data
        self.ale.update_ale()

    def Solve(self):

        self.t.Set(self.t0)
        for step in self(): 
                pass
        
    def PostProcess(self):

        self.prev_def.append(self.ale.deformation.vec.Copy())
        self.prev_vel.append(self.ale.velocity.vec.Copy())
        self.prev_mat_vel.append(self.ale.material_velocity.vec.Copy())    
        if len(self.prev_def)>6:
            self.prev_def.pop(0)
            self.prev_vel.pop(0)
            self.prev_mat_vel.pop(0)
        for pde in self.PDEs:
            pde.Update(self)