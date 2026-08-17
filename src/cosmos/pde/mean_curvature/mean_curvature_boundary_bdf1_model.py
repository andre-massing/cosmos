# Surface mean curvature flow, first-order BDF1 (implicit Euler) time
# discretization, mixed formulation solving simultaneously for the
# displacement update D and the (vector) mean curvature kappa:
#   dD/dt = kappa_param * kappa
#   kappa = Delta_Gamma(X)   (mean curvature = tangential Laplacian of the
#                             identity/position map X)
# using mass-lumped quadrature (ds_lumped) for the reaction-type coupling
# terms between D and kappa. See docs/pde_models.md.

import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.solver import Solver
from cosmos.core.field import InputField, OutputField

class MeanCurvatureBoundaryBDF1Model(BasePDEModel):
    """First-order (BDF1/implicit Euler) mixed finite-element solver for surface
    mean curvature flow, jointly solving for the displacement update and the
    (vector) mean curvature via a mass-lumped mixed formulation."""

    def __init__(self, solver:Solver,
                 model_order:int,
                 domain:str = '.*',
                 name:str = 'MeaCurvatureBoundaryBDF1Model',
                 input_params = {}):
        """
        Args:
            solver: The Solver this model attaches to.
            model_order: Execution order among the solver's attached models.
            domain: Boundary region name(s) the equation is posed on (default: all).
            name: Unique name for this model instance within the solver.
            input_params: Overrides for the default scalar options (see below).

        Recognized input_params keys: `kappa` (scalar flow-speed coefficient
        multiplying the mean curvature vector in the displacement equation).
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

        solver._attach_model(self, self.model_order)

        self.input_params["kappa"] = 1
        self.set_input_params(input_params)

        self.ns = specialcf.normal(self._solver.ngsmesh.dim)
        Ps = Id(self._solver.ngsmesh.dim) - OuterProduct(self.ns, self.ns)

        V1 = VectorH1(self._solver.ngsmesh, order=1,
                    definedon=self.domain)

        fes = CompressCompound(V1*V1)
        self.A = BilinearForm(fes)
        self.F = LinearForm(fes)

        # This model always assembles/solves on the previous configuration:
        # it produces the displacement update to be applied to the mesh, so
        # (unlike e.g. ADRVolumeBDF1Model) it never advances to curr_deformation.
        deform = self._solver.mesh.prev_deformation[-1]

        # Mass-lumped (vertex-based) quadrature rule, used below for the
        # reaction-type coupling terms between D and kappa (their weak mass
        # matrix and coupling terms need to be diagonal/decoupled per node).
        ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])

        ds_lumped = ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = deform)

        # Mixed space: component 0 is the (vector) displacement update D,
        # component 1 is the (vector) mean curvature kappa.
        (trial_D, trial_k), (test_D, test_k) = fes.TnT()
        self.gfu = GridFunction(fes)
        self.gfu_old = GridFunction(fes)
        self.gfu_D, self.gfu_k = self.gfu.components

        self.output_fields["displacement"] = OutputField(self.gfu_D, "displacement", BND)
        self.output_fields["mean_curvature"] = OutputField(self.gfu_k, "mean_curvature", BND)

        rhs_gfu = GridFunction(V1)
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
        # reaction-type coupling dD/dt = kappa_param*kappa (also mass-lumped,
        # below). k-equation: mass term for kappa (mass-lumped) balanced
        # against the weak Laplace-Beltrami operator applied to D.
        self.A += (1/self._solver.time.dt*trial_D*test_D + InnerProduct(trial_k, test_k))*ds_lumped
        self.A += (InnerProduct(grad(trial_D).Trace(), grad(test_k).Trace()))*ds(deformation = deform)
        # Mixed-form coupling: dD/dt = kappa_param * kappa, tested against test_D.
        self.A += (-self.input_params["kappa"]*InnerProduct(trial_k, test_D))*ds_lumped

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
        for the displacement update D and the mean curvature kappa."""

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