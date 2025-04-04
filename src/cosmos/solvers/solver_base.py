# Import necessary libraries
from ngsolve import *
from ngsolve.solvers import Newton
from collections import Counter
from cosmos.pdes.pde_base import BasePDE
from cosmos.utils.tools import params_check
from cosmos.solvers.solver_tools import print_mesh_info
from cosmos.pdes.container import Container
from tqdm import tqdm
import csv

class SolData():

    def __init__(self, mesh, **kwargs):
        
        self.mesh = mesh
        if self.mesh.ne == 0:
            self.domain_markers = list(Counter(self.mesh.GetBoundaries()).keys())
            self.boundary_markers = list(Counter(self.mesh.GetBBoundaries()).keys())
        else:
            self.domain_markers = list(Counter(self.mesh.GetMaterials()).keys())
            self.boundary_markers = list(Counter(self.mesh.GetBoundaries()).keys())

        if len(kwargs)>0:

            if len(kwargs)!=3:
                raise Exception('All time variables must be given: t, T and dt!')
            accepted_keys = [ 't', 'dt', 'T']
            defaults = [Parameter(0), Parameter(0.1), 0.1]
            params_check(params = kwargs, 
                        accepted_keys = accepted_keys, 
                        defaults = defaults)
            
            self.t = kwargs['t']
            self.dt = kwargs['dt']
            self.T = kwargs['T']
            self.iter = 0

class BaseSolver():

    def __init__(self, **kwargs):

        pass

    def AddPDE(self, *args):

        pass

    def Solve(self):

        pass
                    
    def PreProcess(self):

        pass

    def PostProcess(self):

        pass