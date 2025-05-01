# Import necessary libraries
from ngsolve import *
from collections import Counter
from cosmos.pdes.ale import ALE
from cosmos.utils.tools import params_check

class SolverData():

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
            self.prev_dt = []
            self.T = kwargs['T']
            self.iter = 0

class SolverALE():

    def __init__(self, mesh, **kwargs):

        self.mesh = mesh
        self.deformation = GridFunction(Compress(VectorH1(mesh)))
        self.velocity = GridFunction(Compress(VectorH1(mesh)))
        self.prev_d = []
        self.prev_v = []
        self.d_f = None
        self.v_f = None

    def SetMeshDeformation(self, value):
        self.d_f = value

    def SetMeshVelocity(self, value):
        self.v_f = value

    def UpdateMeshDeformation(self):

        if self.d_f:
            if self.mesh.ne == 0:
                self.deformation.Set(self.d_f, definedon = self.mesh.Boundaries('.*'))
            else:
                self.deformation.Set(self.d_f)

    def UpdateMeshVelocity(self):

        if self.v_f:
            if self.mesh.ne == 0:
                self.velocity.Set(self.v_f, definedon = self.mesh.Boundaries('.*'))
            else:
                self.velocity.Set(self.v_f)

    def UpdateALE(self):
        self.UpdateMeshDeformation()
        self.UpdateMeshVelocity()

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