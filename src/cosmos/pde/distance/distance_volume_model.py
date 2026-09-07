"""Distance-to-boundary field -- a geometric utility PDE, not a physics one.

The only ``cosmos.pde`` model in ``cosmos.pde.distance``. Typically
registered with ``ale_type=-1`` (solved once via ``CosmosModel.create_pde``,
not every step -- see the ``pdes_init`` list in ``cosmos.core.model``),
since the geometry it measures against is usually fixed for the run. Its
two outputs, ``sol[0]`` (the distance itself) and ``sol[1]`` (the
normalised gradient, i.e. direction away from the boundary), are meant to
be read by *other* PDE models registered on the same compartment -- e.g. an
ADR model biasing a species' advection velocity to be directed toward or
away from a boundary within a thin layer (``sinh(50*d)/cosh(50*d)``, a
smoothed sign of the distance, is the pattern used for this in the
``article/`` scripts).

Note: the regularisation length scale in ``Initialize()`` (``dt =
specialcf.mesh_size**2``, i.e. tied to the local mesh size) means the
smoothing boundary layer around the zero-set shrinks at the same rate as
the mesh under refinement -- so refining ``h`` alone does not by itself
improve the accuracy of ``sol[0]`` near the boundary; the layer width and
the mesh resolution both shrinking together caps the achievable convergence
rate. See ``tutorials/04_distance_function.py`` for a worked demonstration
of this and of the O(h^2) rate achievable when the regularisation is
decoupled from ``h``.
"""

import logging

logger = logging.getLogger(__name__)

from typing import Optional

from ngsolve import *

from cosmos.core.compartment import CosmosCompartment
from cosmos.core.model import CosmosModel
from cosmos.pde.base import BasePDEModel


class DistanceVolumeModel(BasePDEModel):
    """Computes a smoothed distance function and its gradient on a volume compartment.

    Uses a penalised Poisson solve to obtain a smooth approximation to the
    signed distance from a prescribed zero-boundary, then projects the gradient
    direction via an L2 projection and solves a second Poisson problem for a
    smoother distance. The resulting distance field and normalised-gradient
    (velocity) field are available as output for other PDE models.
    """

    is_bnd = False
    is_vol = True

    def __init__(
        self,
        name: str = "DistanceVolumeModel",
        model: Optional[CosmosModel] = None,
        compartment: Optional[CosmosCompartment] = None,
        **kwargs,
    ):

        super().__init__(name=name, model=model, compartment=compartment)

        if "zero_bnd" not in kwargs.keys():
            raise ValueError(
                "The argument -zero_bnd- has to be passed at initialization for DistanceVolumeModel"
            )
        else:
            self.params["zero_bnd"] = kwargs["zero_bnd"]

        self.params["fes_order"] = 2
        self.params["subdivision"] = 0

        self.fes = H1(
            model.parentmesh,
            order=self.params["fes_order"],
            definedon=compartment.domain,
            dirichlet=self.params["zero_bnd"],
        )
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)

        self.fes2 = VectorH1(
            self.model.parentmesh,
            order=self.params["fes_order"] - 1,
            definedon=self.compartment.domain,
            dirichlet=self.model.parentmesh.Boundaries(self.params["zero_bnd"]),
        )
        self.gfu2 = GridFunction(self.fes2)

        self.sol = [self.gfu, self.gfu2]

        self.vtk_gfu = self.sol
        self.vtk_names = [self.name + "_distance", self.name + "_velocity"]

    def Initialize(self):

        delta = specialcf.mesh_size**2

        fes1 = H1(
            self.model.parentmesh,
            order=self.params["fes_order"],
            definedon=self.compartment.domain,
            dirichlet=self.params["zero_bnd"],
        )

        u1, v1 = fes1.TnT()
        self.A1 = BilinearForm(fes1)
        self.A1 += (u1 * v1 + delta * grad(u1) * grad(v1)) * dx(deformation=self.model.ale.Y)
        self.A1.Assemble()
        self.invA1 = self.A1.mat.Inverse(freedofs=fes1.FreeDofs())
        self.gfu1 = GridFunction(fes1)

        u2, v2 = self.fes2.TnT()
        self.A2 = BilinearForm(self.fes2)
        self.A2 += u2 * v2 * dx(deformation=self.model.ale.Y)
        self.A2.Assemble()
        self.invA2 = self.A2.mat.Inverse(freedofs=self.fes2.FreeDofs())
        self.F2 = LinearForm(self.fes2)
        self.F2 += Normalize(grad(self.gfu1)) * v2 * dx(deformation=self.model.ale.Y)

        u3, v3 = self.fes.TnT()
        self.A3 = BilinearForm(self.fes)
        self.A3 += grad(u3) * grad(v3) * dx(deformation=self.model.ale.Y)
        self.A3.Assemble()
        self.invA3 = self.A3.mat.Inverse(freedofs=self.fes.FreeDofs())
        self.F3 = LinearForm(self.fes)
        self.F3 += Trace(Grad(self.gfu2)) * v3 * dx(deformation=self.model.ale.Y)

        logger.info(
            f"[Cosmos] PDE '{self.name}' ({type(self).__name__}) initialized "
            f"on compartment '{self.compartment.name}'"
        )

    def PreProcess(self):

        pass

    def Solve(self):

        self.gfu.vec.data[:] = 0

        self.A1.Assemble()
        self.invA1.Update()
        self.gfu1.Set(1, definedon=self.model.parentmesh.Boundaries(self.params["zero_bnd"]))
        res1 = -1 * self.A1.mat * self.gfu1.vec
        self.gfu1.vec.data += self.invA1 * res1

        self.A2.Assemble()
        self.invA2.Update()
        self.F2.Assemble()
        self.gfu2.Set(
            specialcf.normal(self.model.dim),
            definedon=self.model.parentmesh.Boundaries(self.params["zero_bnd"]),
        )
        res2 = self.F2.vec - self.A2.mat * self.gfu2.vec
        self.gfu2.vec.data += self.invA2 * res2

        self.A3.Assemble()
        self.invA3.Update()
        self.F3.Assemble()
        self.gfu.vec.data += self.invA3 * self.F3.vec

    def PostProcess(self):

        pass
