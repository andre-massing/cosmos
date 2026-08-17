# Surface mean curvature flow, first-order BDF1 (implicit Euler) time
# discretization, same mixed formulation as MeanCurvatureBoundaryBDF1Model
# (solving jointly for the displacement update D and the mean curvature
# kappa via mass-lumped coupling terms), but with a third unknown -- the
# facet-normal derivative jump of the curvature, in a NormalFacetSurface
# space -- and a jump-penalization stabilization term (coefficient
# `stabilization`) to control spurious oscillations of the discrete
# curvature. Only implemented for 3D ambient space. See docs/pde_models.md.

import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.solver import Solver
from cosmos.core.field import InputField, OutputField

class MeanCurvatureBoundaryStabBDF1Model(BasePDEModel):
    """First-order (BDF1/implicit Euler) mixed finite-element solver for surface
    mean curvature flow with an added facet-jump stabilization term on the
    discrete curvature, to suppress spurious oscillations. 3D ambient space
    only (raises for a 2D mesh, i.e. a 1D-manifold curve)."""

    def __init__(self, solver:Solver,
                 model_order:int,
                 domain:str = '.*',
                 name:str = 'MeanCurvatureBoundaryStabBDF1Model',
                 input_params = {}):
        """
        Args:
            solver: The Solver this model attaches to.
            model_order: Execution order among the solver's attached models.
            domain: Boundary region name(s) the equation is posed on (default: all).
            name: Unique name for this model instance within the solver.
            input_params: Overrides for the default scalar options (see below).

        Recognized input_params keys: `stabilization` (coefficient of the
        facet-normal jump-penalization term added to the curvature equation).
        Raises an Exception if the mesh is 2D (a 1D-manifold curve), since
        stabilization is only implemented for 3D ambient space.
        """

        super().__init__()

        self.VorB = BND
        self.name = name
        self._solver = solver
        self.model_order = model_order

        if (domain in solver.mesh.bnd_markers) or domain == '.*':
            self.domain = solver.ngsmesh.Boundaries(domain)
        else:
            logger.error('The domain specified for the Model ', self.name, ' does not exist')

        # ngsmesh.dim is the ambient space dimension; dim==2 means the boundary
        # is a 1D-manifold (a curve), for which the facet-jump stabilization
        # below is not implemented.
        if self._solver.ngsmesh.dim == 2:
            raise Exception('Stabilization not implemented for 1D manifolds!')

        solver._attach_model(self, self.model_order)

        self.input_params["stabilization"] = 0.01

        self.set_input_params(input_params)

        h = self.cfg.h
        tE = specialcf.tangential(self._solver.ngsmesh.dim)
        self.ns = specialcf.normal(self._solver.ngsmesh.dim)
        Ps = Id(self._solver.ngsmesh.dim) - OuterProduct(self.ns, self.ns)
        # nE is the facet (edge) conormal used in the jump-stabilization term
        # below. Note: the dim==2 branch is effectively unreachable, since the
        # constructor already raises above for a 2D mesh.
        if self._solver.ngsmesh.dim == 2:
            nE = tE
            tEc = CF((-self.ns[1], self.ns[0]))
        else:
            nE = Cross(self.ns, tE)


        V1 = VectorH1(self._solver.ngsmesh, order=1,
                    definedon=self.domain)
        V2 = VectorH1(self._solver.ngsmesh, order=1,definedon=self.domain)
        V3 = H1(self._solver.ngsmesh, order=1,definedon=self.domain)
        # if self._solver.ngsmesh.dim == 2:
        #     dV = H1(self._solver.ngsmesh, order=1,definedon = self.domain)
        # elif self._solver.ngsmesh.dim == 3:
        # NormalFacetSurface: auxiliary space for the facet-normal derivative
        # jump unknown dkappa/dn used by the stabilization term below.
        dV = NormalFacetSurface(self._solver.ngsmesh, order=0, definedon = self.domain)

        fes = CompressCompound(V1*V2*dV)
        self.A = BilinearForm(fes)
        self.F = LinearForm(fes)

        # This model always assembles/solves on the previous configuration:
        # it produces the displacement update to be applied to the mesh, so
        # (unlike e.g. ADRVolumeBDF1Model) it never advances to curr_deformation.
        deform = self._solver.mesh.prev_deformation[-1]

        # Mass-lumped (vertex-based) quadrature rule, used below for the
        # reaction-type coupling terms between D and kappa.
        ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])

        ds_lumped = ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = deform)
        ds_el_lumped = ds(element_boundary=True, intrules = { SEGM : ir_segm, TRIG: ir_trig })

        # Mixed space: component 0 is the (vector) displacement update D,
        # component 1 is the (vector) mean curvature kappa, component 2 is the
        # auxiliary facet-normal-derivative-jump unknown dkappa (dV space).
        (trial_D, trial_k, trial_dk), (test_D, test_k, test_dk) = fes.TnT()
        self.gfu = GridFunction(fes)
        self.gfu_old = GridFunction(fes)
        self.gfu_D, self.gfu_k, _ = self.gfu.components

        self.output_fields["displacement"] = OutputField(self.gfu_D, "displacement", BND)
        self.output_fields["mean_curvature"] = OutputField(self.gfu_k, "mean_curvature", BND)


        rhs_gfu = GridFunction(V2)
        self.input_fields["rhs"] = InputField(rhs_gfu, CF((0,)*self._solver.ngsmesh.dim), "rhs", self._solver.ngsmesh.Boundaries('.*'))

        # X0 stores the identity/position map at construction time; displacement_tot
        # is reserved for tracking the accumulated total displacement.
        self.X0 = GridFunction(V1)
        if self._solver.ngsmesh.dim == 2:
            self.X0.Set(CF((x,y)), dual = True, definedon=self.domain)
        elif self._solver.ngsmesh.dim == 3:
            self.X0.Set(CF((x, y, z)), dual = True, definedon=self.domain)
        self.displacement_tot = GridFunction(V1)

        # D-equation: BDF1 time derivative of D (mass-lumped), plus the
        # reaction-type coupling dD/dt = kappa (also mass-lumped, below).
        # k-equation: mass term for kappa (mass-lumped) balanced against the
        # weak Laplace-Beltrami operator applied to D.
        self.A += (1/self._solver.time.dt*trial_D*test_D + InnerProduct(trial_k, test_k))*ds_lumped
        self.A += (InnerProduct(grad(trial_D).Trace(), grad(test_k).Trace()))*ds(deformation = deform)
        # Mixed-form coupling: dD/dt = kappa, tested against test_D.
        self.A += (-InnerProduct(trial_k, test_D))*ds_lumped
        # Facet jump-penalization stabilization: penalizes the mismatch between
        # the facet-normal derivative of the discrete curvature evaluated from
        # the (discontinuous, element-local) kappa field, trial_k.Trace().Deriv()*nE,
        # and the auxiliary dual unknown trial_dk (living in the NormalFacetSurface
        # space, continuous across facets) meant to represent the same quantity;
        # penalizing their jump suppresses spurious oscillations of kappa.
        jump_dkappadn = (trial_k.Trace().Deriv()*nE-trial_dk.Trace())
        jump_detadn = (test_k.Trace().Deriv()*nE-test_dk.Trace())
        self.A += (self.input_params["stabilization"]*h*InnerProduct(jump_dkappadn,jump_detadn))*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, 
                                                                                                    element_boundary=True, deformation = deform)

        # RHS for the k-equation: since grad(X0) = Ps (tangential projector) for
        # the identity map, this term supplies the "source" from the current
        # geometry after integrating the weak Laplace-Beltrami form by parts.
        self.F += (-InnerProduct(Ps, grad(test_k).Trace()))*ds(deformation = deform)
        self.F += (InnerProduct(rhs_gfu, test_D))*ds_lumped

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = fes.FreeDofs())

    def PreProcess(self):
        """Snapshot the current solution as the previous time level."""

        self.gfu_old.vec.data = self.gfu.vec.data

    def Solve(self):
        """Refresh input fields, re-assemble and solve the coupled linear system
        for the displacement update D, the mean curvature kappa, and the
        auxiliary facet-jump unknown."""

        self.update_input_fields()

        self.A.Assemble()
        self.invA.Update()
        self.F.Assemble()

        self.gfu.vec.data = self.invA*self.F.vec

    def PostProcess(self):
        """No auxiliary state to advance for this model."""
        pass

    @property
    def displacement(self):
        """The current displacement-update GridFunction (D)."""
        return self.gfu_D

    # Note: the setter below is named `phase`, not `displacement` -- a
    # leftover from copy-pasting the Cahn-Hilliard model's property pattern.
    # Kept as-is (calling `self.displacement = cf` will NOT invoke this
    # setter); do not rename without checking all call sites.
    @displacement.setter
    def phase(self, cf):
        self.gfu_D.Set(cf, definedon = self.domain)

    @property
    def mean_curvature(self):
        """The current (vector) mean curvature GridFunction (kappa)."""
        return self.gfu_k

    # Note: same leftover naming quirk as above -- the setter is named
    # `potential` rather than `mean_curvature`.
    @mean_curvature.setter
    def potential(self, cf):
        self.gfu_k.Set(cf, definedon = self.domain)