# Wraps an NGSolve Mesh with the deformation-history bookkeeping needed for
# evolving-domain (moving-mesh) simulations.

import logging
logger = logging.getLogger(__name__)

from collections import Counter
from ngsolve import *
from cosmos.config.parameters import get_config


class SolverMesh:
    """Wraps an NGSolve mesh and tracks its deformation state over time.

    Supports both pure surface meshes (``mesh.ne == 0``, a codimension-1 manifold with
    no volume elements, e.g. a triangulated sphere embedded in 3D) and bulk/volume
    meshes uniformly, exposing region-name sets and a deformation-space GridFunction
    used by every PDE model to integrate on the deformed configuration.
    """

    def __init__(self, mesh:Mesh):

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

        # Region names discovered from the mesh: materials (volume), boundaries
        # (codimension-1), and "bboundaries" (codimension-2, e.g. edges of a surface).
        self.vol_markers = set(Counter(self.mesh.GetMaterials()).keys())
        self.bnd_markers = set(Counter(self.mesh.GetBoundaries()).keys())
        self.bbnd_markers = set(Counter(self.mesh.GetBBoundaries()).keys())

        # A mesh with zero volume elements is a pure surface (manifold) mesh; the
        # deformation space is then built over the whole boundary rather than the
        # whole volume.
        self.is_bnd = False
        if self.ne == 0:
            self.is_bnd = True
            self.domain = self.mesh.Boundaries('.*')
            self.V = VectorH1(self.mesh, order=self.mesh.GetCurveOrder(), definedon = self.domain)
        else:
            self.domain = self.mesh.Materials('.*')
            self.V = VectorH1(self.mesh, order=self.mesh.GetCurveOrder(), definedon = self.domain)
        _deformation = GridFunction(self.V)
        self.mesh.SetDeformation(_deformation)
        # Ring buffer of the last `buffer` committed deformations (needed by
        # multi-step time discretizations such as BDF2), plus the deformation
        # currently being computed for the in-progress step.
        self.prev_deformation = [GridFunction(self.V)]*self.buffer
        self.curr_deformation = GridFunction(self.V)

        logger.debug('Mesh has been initialized correctly')

    def update_state(self):
        """Commit `curr_deformation` as the mesh's new deformation and shift the
        `prev_deformation` history buffer (called once a time step is accepted,
        typically from ALEModel.PostProcess)."""

        for i in range(self.buffer - 1):
            self.prev_deformation[i].vec.data = self.prev_deformation[i+1].vec.data
        self.prev_deformation[-1].vec.data = self.curr_deformation.vec.data
        self.mesh.deformation.vec.data = self.curr_deformation.vec.data

    def get_state(self):
        """Return a dict snapshot of the current and previous deformations."""
        mesh_state = {
            "deformation": self.curr_deformation,
            "prev_def": self.prev_deformation
        }
        return mesh_state

    def advance_mesh(self):
        """Temporarily apply `curr_deformation` to the underlying NGSolve mesh so
        forms can be assembled on the tentative next-step configuration."""
        self.mesh.deformation.vec.data = self.curr_deformation.vec.data

    def reset_mesh(self):
        """Restore the last committed deformation, undoing `advance_mesh()`."""
        self.mesh.deformation.vec.data = self.prev_deformation[-1].vec.data

    def print_info(self):
        """Print a human-readable summary of the mesh's regions and element counts."""

        print('The mesh has dimension: ', self.dim)
        print('The mesh has the following codimension-0 domains:')
        print(self.vol_markers)
        print('The mesh has the following codimension-1 domains:')
        print(self.bnd_markers)
        print('The mesh has the following codimension-2 domains:')
        print(self.vol_markers)
        print('Mesh characteristics are:')
        print('\t - number of elements: ', self.ne)
        print('\t - number of vertices: ', self.nv)
        print('\t - number of facets: ', self.nfacet)
        print('\t - number of faces: ', self.nface)
        print('\t - number of edges: ', self.nedge)
    
    @property
    def deformation(self):
        """The deformation GridFunction currently being computed for this step."""
        return self.curr_deformation



    