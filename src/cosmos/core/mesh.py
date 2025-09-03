import logging
logger = logging.getLogger(__name__)

from collections import Counter
from ngsolve import *
from cosmos.config.parameters import get_config

class SolverMesh:

    def __init__(self, mesh:Mesh, dirichlet_bnd = ''):

        self.mesh = mesh
        self.ne = self.mesh.ne
        self.nedge = self.mesh.nedge
        self.nface = self.mesh.nface
        self.nfacet = self.mesh.nfacet
        self.ngmesh = self.mesh.ngmesh
        self.nv = self.mesh.nv
        self.dim = self.mesh.dim
        self.order = self.mesh.GetCurveOrder()
        self.buffer = get_config().buffer

        self.vol_markers = list(Counter(self.mesh.GetMaterials()).keys())
        self.bnd_markers = list(Counter(self.mesh.GetBoundaries()).keys())
        self.bbnd_markers = list(Counter(self.mesh.GetBBoundaries()).keys())

        self.is_bnd = False
        if self.ne == 0:
            self.is_bnd = True
            self.domain = self.mesh.Boundaries('.*')
            self.V = VectorH1(self.mesh, order=self.mesh.GetCurveOrder(), definedon = self.domain)
        else:
            self.domain = self.mesh.Materials('.*')
            self.V = VectorH1(self.mesh, order=self.mesh.GetCurveOrder(), definedon = self.domain, dirichlet = dirichlet_bnd)
        _deformation = GridFunction(self.V)
        self.mesh.SetDeformation(_deformation)
        self.prev_deformation = [GridFunction(self.V)]*self.buffer
        self.curr_deformation = GridFunction(self.V)

        logger.debug('Mesh has been initialized correctly')

    def update_state(self):

        for i in range(self.buffer - 1):
            self.prev_deformation[i].vec.data = self.prev_deformation[i+1].vec.data
        self.prev_deformation[-1].vec.data = self.curr_deformation.vec.data
        self.mesh.deformation.vec.data = self.curr_deformation.vec.data

    def get_state(self):
        mesh_state = {
            "deformation": self.curr_deformation,
            "prev_def": self.prev_deformation
        }
        return mesh_state
    
    def advance_mesh(self):
        self.mesh.deformation.vec.data = self.curr_deformation.vec.data

    def reset_mesh(self):
        self.mesh.deformation.vec.data = self.prev_deformation[-1].vec.data
    
    @property
    def deformation(self):
        return self.curr_deformation



    