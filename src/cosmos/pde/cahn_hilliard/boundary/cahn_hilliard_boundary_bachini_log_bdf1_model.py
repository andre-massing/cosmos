# Surface (boundary) two-field Cahn-Hilliard model, first-order BDF1
# (implicit Euler) time discretization, following Bachini et al.'s surface
# discretization of the mixed Cahn-Hilliard system using tangential
# (surface-gradient) operators, with a LOGARITHMIC Flory-Huggins-style
# double-well potential:
#   du/dt + div_Gamma(b u) - div_Gamma(M grad_Gamma w) = rhs_u
#   w - sigma*epsilon*Delta_Gamma(u) = sigma/epsilon*dW(u_old)
# The potential is evaluated explicitly at u_old (no ddW-based linearization
# here, mirroring the volume log model). See docs/pde_models.md.

import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.solver import Solver
from cosmos.core.field import InputField, OutputField
from cosmos.core.utils import MandBP

class CahnHilliardBoundaryBachiniLogBDF1Model(BasePDEModel):
    """First-order (BDF1/implicit Euler) mixed Cahn-Hilliard solver on a surface
    (boundary) domain, following Bachini et al.'s tangential-operator
    discretization, with a logarithmic (Flory-Huggins-style) double-well
    potential evaluated explicitly at the previous time step."""

    def __init__(self, solver:Solver,
                 model_order:int,
                 domain:str = '.*',
                 name:str = 'CahnHilliardBoundaryBachiniLogBDF1Model',
                 input_params = {}):
        """
        Args:
            solver: The Solver this model attaches to.
            model_order: Execution order among the solver's attached models.
            domain: Boundary region name(s) the equation is posed on (default: all).
            name: Unique name for this model instance within the solver.
            input_params: Overrides for the default scalar options (see below).

        Recognized input_params keys: `periodic`, `mass_preserving`, `bounds`
        (`[low, high]`), `fes_order` (bound/mass preservation only implemented
        for `fes_order == 1`), `u0`/`w0` (initial conditions for phase `u` and
        potential `w`), `M` (mobility), `epsilon` (interface-width parameter),
        `sigma` (surface tension), `Neu_bnd_phase`/`Neu_bnd_potential`
        (Neumann boundary-of-the-surface region names for the phase/potential
        equations).
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

        n = specialcf.normal(self._solver.ngsmesh.dim)
        tE = specialcf.tangential(self._solver.ngsmesh.dim)
        Ps = Id(self._solver.ngsmesh.dim) - OuterProduct(n, n)
        # facet_space/nE are only used to mark and orient the co-dimension-2
        # Neumann boundary-of-the-surface (a curve in 3D, a point in 2D) further
        # below; nE is the outward conormal within the surface at that boundary.
        if self._solver.ngsmesh.dim == 2:
            facet_space = H1(self._solver.ngsmesh, order = 1, 
                                            definedon = self.domain)
            nE = specialcf.tangential(self._solver.ngsmesh.dim)
        else:
            facet_space = FacetSurface(self._solver.ngsmesh, order = 0)
            nE = Cross(n, tE)

        self.input_params["periodic"] = False
        self.input_params["mass_preserving"] = False
        self.input_params["bounds"] = None
        self.input_params["fes_order"] = 1
        self.input_params["u0"] = CF(0)
        self.input_params["w0"] = CF(0)
        self.input_params["M"] = 1
        self.input_params["epsilon"] = 1
        self.input_params["sigma"] = 1
        self.input_params["Neu_bnd_phase"] = ''
        self.input_params["Neu_bnd_potential"] = ''

        self.set_input_params(input_params)
        
        _fes = H1(self._solver.ngsmesh, order = self.input_params["fes_order"], 
                                            definedon = self.domain)
        if self.input_params["periodic"]:
            fes = Compress(Periodic(_fes))*Compress(Periodic(_fes))
            fes_vector = Compress(Periodic(VectorH1(self._solver.ngsmesh, order = self.input_params["fes_order"], 
                                            definedon = self.domain)))
        else:
            fes = _fes*_fes
            fes_vector = Compress(VectorH1(self._solver.ngsmesh, order = self.input_params["fes_order"], 
                                   definedon = self.domain))
          
        # Mixed space: component 0 is the phase field u, component 1 is the
        # chemical potential w, both restricted to the surface `self.domain`.
        (trial_u, trial_w), (test_u, test_w) = fes.TnT()
        self.A = BilinearForm(fes)
        self.F = LinearForm(fes)
        
        self.gfu = GridFunction(fes)
        self.gfu_u, self.gfu_w = self.gfu.components
        self.gfu_old = GridFunction(fes)
        self.gfu_u_old, self.gfu_w_old = self.gfu_old.components
        self.gfu_u.Set(self.input_params['u0'], dual = True, definedon = self.domain)
        self.output_fields["phase"] = OutputField(self.gfu_u, "phase", BND)
        self.gfu_w.Set(self.input_params['w0'], dual = True, definedon = self.domain)
        self.output_fields["potential"] = OutputField(self.gfu_w, "potential", BND)

        deform = self._solver.mesh.prev_deformation[-1]

        # Creating GridFunctions for the Fields
        b_gfu = GridFunction(fes_vector)
        rhs_u_gfu = GridFunction(_fes)
        rhs_w_gfu = GridFunction(_fes)
        grad_phase_bnd_gfu = GridFunction(fes_vector)
        grad_potential_bnd_gfu = GridFunction(fes_vector)
        self.input_fields["b"] = InputField(b_gfu, CF((0,)*solver.mesh.dim), "b", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["rhs_u"] = InputField(rhs_u_gfu, CF(0), "rhs_u", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["rhs_w"] = InputField(rhs_w_gfu, CF(0), "rhs_w", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["grad_phase_bnd"] = InputField(grad_phase_bnd_gfu, CF((0,)*self._solver.ngsmesh.dim), "grad_phase_bnd", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["grad_potential_bnd"] = InputField(grad_potential_bnd_gfu, CF((0,)*self._solver.ngsmesh.dim), "grad_potential_bnd", self._solver.ngsmesh.Boundaries('.*'))

        # Logarithmic double-well potential derivative dW(u) =
        # 0.25*(log10(1+u) - log10(1-u)) - u, evaluated explicitly at the
        # previous time level u_old (see F below); note this uses base-10 log
        # via the self.log10 helper method defined near the end of this class.
        def dW(phase):
            return 0.25*(self.log10(1+phase)-self.log10(1-phase))-phase 

        # BDF1 time derivative plus tangential advection of u.
        self.A += 1/self._solver.time.dt*trial_u*test_u*ds(deformation = deform)
        self.A += b_gfu*grad(trial_u).Trace()*test_u*ds(deformation = deform)
        # Mixed-form coupling: -div_Gamma(M grad_Gamma w) (u-equation) tested
        # weakly as M*grad_Gamma(w).grad_Gamma(test_u) (surface gradients via
        # .Trace()).
        self.A += self.input_params["M"]*grad(trial_w).Trace()*grad(test_u).Trace()*ds(deformation = deform)
        self.A += trial_w*test_w*ds(deformation = deform)
        # w-equation: -sigma*epsilon*Delta_Gamma(u) (integrated by parts); the
        # potential term dW(u_old) is added explicitly to F below (no implicit
        # linearization for the logarithmic potential in this variant).
        self.A += -self.input_params["sigma"]*self.input_params["epsilon"]*grad(trial_u).Trace()*grad(test_w).Trace()*ds(deformation = deform)

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = fes.FreeDofs())

        self.F += rhs_u_gfu*test_u*ds(deformation = deform)
        self.F += rhs_w_gfu*test_w*ds(deformation = deform)

        # BDF1 time derivative term plus the explicit potential source dW(u_old).
        self.F += 1/self._solver.time.dt*self.gfu_u_old*test_u*ds(deformation = deform)
        self.F += self.input_params["sigma"]/self.input_params["epsilon"]*dW(self.gfu_u_old)*test_w*ds(deformation = deform)

        if self.input_params["Neu_bnd_phase"]:
            # Neumann flux for the phase equation, imposed weakly on the
            # boundary-of-the-surface curve/point selected via Neu_bnd_phase;
            # neu_bnd_gfu_phase is an indicator restricting the term to it.
            neu_bnd_gfu_phase = GridFunction(facet_space)
            neu_bnd_gfu_phase.Set(1, definedon = self._solver.ngsmesh.BBoundaries(self.input_params['Neu_bnd_phase']))
            self.F += -1*neu_bnd_gfu_phase*self.input_params["sigma"]*self.input_params["epsilon"]*grad_phase_bnd_gfu*nE*test_w*ds(element_boundary = True, deformation = deform)
        if self.input_params["Neu_bnd_potential"]:
            # Neumann flux for the potential equation, same mechanism.
            neu_bnd_gfu_potential = GridFunction(facet_space)
            neu_bnd_gfu_potential.Set(1, definedon = self._solver.ngsmesh.BBoundaries(self.input_params['Neu_bnd_potential']))
            self.F += neu_bnd_gfu_potential*self.input_params["M"]*grad_potential_bnd_gfu*nE*test_u*ds(element_boundary=True, deformation = deform)

        if self.input_params["mass_preserving"]:
            # Mass-lumped (vertex-based) surface quadrature rules used to build
            # a diagonal mass matrix whose entries serve as quadrature weights
            # for MandBP.
            ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = {  SEGM : ir_segm, TRIG : ir_trig })
            self.Amp = BilinearForm(self.gfu_u.space, symmetric = True)
            u, v = self.gfu_u.space.TnT()
            self.Amp += u*v*ds_lumped
            self.Amp.Assemble()
            rows,cols,vals = self.Amp.mat.COO()
            self.weights = sp.csr_matrix((vals,(rows,cols))).diagonal()

    def PreProcess(self):
        """Snapshot the current solution as the previous time level; on the first
        iteration, validate fes_order for bound/mass preservation and, if mass
        preservation is requested, record the target mass."""

        self.gfu_old.vec.data = self.gfu.vec.data

        if self._solver.time.iter == 0:

            if self.input_params["mass_preserving"] and self.input_params["fes_order"]>1:
                raise Exception('Mass preservation not yet implemented for fes_order>1')
            if self.input_params["bounds"] and self.input_params["fes_order"]>1:
                raise Exception('Bounds preservation not yet implemented for fes_order>1')

            if self.input_params["mass_preserving"]:
                gfu0_vec = self.gfu_u.vec.Copy().FV().NumPy()
                self.mass0 = np.sum(self.weights*gfu0_vec)

    def Solve(self):
        """Refresh input fields, re-assemble and solve the linear system for this
        step, then optionally apply bound/mass-preservation post-processing to
        the phase."""

        self.update_input_fields()

        self.A.Assemble()
        self.invA.Update()
        self.F.Assemble()

        self.gfu.vec.data = self.invA*self.F.vec
        
        if self.input_params["bounds"] and not self.input_params["mass_preserving"]:

            gfu_vec = self.gfu_u.vec.Copy().FV().NumPy()
            gfu_new = MandBP(gfu_vec, BP = self.input_params["bounds"])
            self.gfu_u.vec.data = gfu_new

        elif self.input_params["mass_preserving"]:

            if hasattr(self._solver.time, 'dt'):
                dt = self._solver.time.dt.Get()
            else:
                logger.error('A time-dependent simulation is needed to impose conservative mass')

            self.Amp.Assemble()
            rows,cols,vals = self.Amp.mat.COO()
            weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
            gfu_vec = self.gfu_u.vec.Copy().FV().NumPy()

            if self.input_params["bounds"]:
                BP = self.input_params["bounds"]
            else:
                BP = [-np.inf, np.inf]

            gfu_new = MandBP(gfu_vec, weights=weights, BP=BP,
                                MP=self.input_params["mass_preserving"], mass0=self.mass0, dt = dt)

            self.gfu_u.vec.data = gfu_new

    def PostProcess(self):
        """No auxiliary state to advance for this model."""

        pass

    def log10(self, a):
        """Base-10 logarithm of a coefficient function, via natural log."""
        return log(a)/log(10)

    @property
    def phase(self):
        """The current phase field GridFunction (u)."""
        return self.gfu_u

    @phase.setter
    def phase(self, cf):
        """Overwrite the current phase field by interpolating a coefficient."""
        self.gfu_u.Set(cf, definedon = self.domain)

    @property
    def potential(self):
        """The current chemical potential GridFunction (w)."""
        return self.gfu_w

    @potential.setter
    def potential(self, cf):
        """Overwrite the current chemical potential by interpolating a coefficient."""
        self.gfu_w.Set(cf, definedon = self.domain)

    @property
    def energy(self):
        """Total logarithmic (Flory-Huggins) free energy, combining the entropic
        mixing term, the double-well term, and the gradient (interface) term,
        integrated over the boundary domain."""
        energy = Integrate(self.input_params['sigma']*(0.25/self.input_params['epsilon']*((1-self.phase)*self.log10(1-self.phase) 
                            + (1+self.phase)*self.log10(1+self.phase)) + 0.5/self.input_params['epsilon']*(1-self.phase**2)
                           + self.input_params['epsilon']/2*Norm(grad(self.phase).Trace())**2), 
                           self._solver.ngsmesh, VOL_or_BND = BND)
        return energy

