# Surface Willmore (Helfrich bending) flow, first-order BDF1 in time,
# Area-AND-Volume-Preserving (APVP) variant. Solves the same mixed (D, Y)
# system as the baseline model (willmore_boundary_bdf1_model.py; see that
# file for the physics of the split L2-gradient flow of E = 0.5*
# kappa_elastic*integral((H-H0)^2)), but augments it with TWO scalar Lagrange
# multipliers: `lambda_h` (as in the area-preserving variant,
# willmore_boundary_ap_bdf1_model.py) enforcing conservation of surface area,
# and `omega_h` (as in the volume-preserving variant,
# willmore_boundary_vp_bdf1_model.py) enforcing conservation of enclosed
# volume, applied simultaneously. Because both constraints are coupled and
# nonlinear, Solve() runs a fixed-point loop that at each iteration solves the
# 2x2 linear system for (lambda_h, omega_h) implied by linearizing the two
# constraint (Rayleigh-quotient-like) equations about the current iterate,
# then re-solves the mixed (D, Y) system with the updated multipliers, until
# both multipliers stop changing (or raise after `maxiter` iterations without
# convergence). Not compatible with clamped_bnd. See docs/pde_models.md.

import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.solver import Solver
from cosmos.core.field import InputField, OutputField
from ngsolve.webgui import Draw

class WillmoreBoundaryAPVPBDF1Model(BasePDEModel):
    """Area- and volume-preserving BDF1 mixed FE solver for Willmore
    (Helfrich bending) flow of a boundary (surface) mesh.

    Identical mixed (D, Y) formulation to WillmoreBoundaryBDF1Model, plus two
    scalar NumberSpace unknowns: `lambda_h` (area constraint, reacting on the
    surface Laplace-Beltrami operator applied to D, as in the AP variant) and
    `omega_h` (volume constraint, reacting along the surface normal, as in
    the VP variant). Because the two constraint equations are coupled through
    the shared solution, Solve() solves a 2x2 linear system for
    (lambda_h, omega_h) at each fixed-point iteration (rather than the single
    scalar update used by the AP/VP variants individually), then re-solves
    the mixed system, repeating until both multipliers converge. Raises at
    construction time if `clamped_bnd` is set (not yet supported together
    with area/volume preservation).
    """

    def __init__(self, solver:Solver,
                 model_order:int,
                 domain:str = '.*',
                 name:str = 'WillmoreBoundaryAPVPBDF1Model',
                 input_params = {}):
        """
        Args:
            solver: The Solver this model attaches to.
            model_order: Execution order among the solver's attached models.
            domain: Boundary region name(s) the equation is posed on (default: all).
            name: Unique name for this model instance within the solver.
            input_params: Overrides for the default scalar options (see below).

        Recognized input_params keys:
            clamped_bnd: Must be left at '' — area-and-volume-preserving
                Willmore flow with a clamped boundary is not yet implemented
                and raises.
            clamped_conormal: Prescribed co-normal vector (unused unless a
                future implementation enables clamped_bnd here).
            autoupdate: If True, re-solve the auxiliary mean-curvature
                projection (A_mc/F_mc) every step instead of only at iter 0.
        """

        super().__init__()

        self.VorB = BND
        self.name = name
        self._solver = solver
        self.model_order = model_order

        if (domain in solver.mesh.bnd_markers) or domain == '.*':
            self.domain = solver.ngsmesh.Boundaries(domain)
        else:
            raise Exception('The domain specified for the Model ' + self.name + ' does not exist')

        solver._attach_model(self, self.model_order)

        self.input_params["clamped_bnd"] = ''
        self.input_params["clamped_conormal"] = CF((0,)*self._solver.ngsmesh.dim)
        self.input_params["autoupdate"] = False

        self.set_input_params(input_params)

        if self.input_params["clamped_bnd"]!='':
            raise('Area and Volume preserving algorithm not yet implemented for Willmore flow with clamped boundaries')
        
        if (self.input_params["clamped_bnd"]!='') and \
            (self.input_params["clamped_bnd"] not in self._solver.mesh.bbnd_markers):
            raise ValueError('Clamped boundary conditions are imposed on non-existing BBoundary')

        h = self.cfg.h
        # Surface differential geometry quantities: tE/nE are the (edge)
        # tangent and co-normal of the boundary of the surface, ns is the
        # surface unit normal, and Ps is the tangential projector used
        # throughout to restrict gradients to the surface tangent plane.
        tE = specialcf.tangential(self._solver.ngsmesh.dim)
        self.ns = specialcf.normal(self._solver.ngsmesh.dim)
        self.Ps = Id(self._solver.ngsmesh.dim) - OuterProduct(self.ns, self.ns)
        if self._solver.ngsmesh.dim == 2:
            nE = tE
            tEc = CF((-self.ns[1], self.ns[0]))
        else:
            nE = Cross(self.ns, tE)

        # TODO: Check is this can actually be merged 
        # V1: displacement space for D (unused clamped_bnd branch kept for
        # parity with the sibling models; raised above before reaching here).
        if self.input_params["clamped_bnd"]:
            V1 = VectorH1(self._solver.ngsmesh, order=1, definedon=self.domain,
                    dirichlet_bbnd = self._solver.ngsmesh.BBoundaries(self.input_params["clamped_bnd"]))
        else:
            V1 = VectorH1(self._solver.ngsmesh, order=1,
                        definedon=self.domain)
        # V2: space for the auxiliary variable Y (and the mean curvature
        # gfu_k). V3: scalar coefficient space.
        V2 = VectorH1(self._solver.ngsmesh, order=1,definedon=self.domain)
        V3 = H1(self._solver.ngsmesh, order=1, definedon=self.domain)
        
        # V4: single-DOF NumberSpace shared by the two scalar Lagrange
        # multipliers: omega_h (volume) and lambda_h (area); see Solve().
        V4 = NumberSpace(self._solver.ngsmesh, definedon=self.domain)
        self.omega_h = GridFunction(V4)
        self.lambda_h = GridFunction(V4)

        # pre_normal/normal hold a P1 interpolation of the (unit) surface
        # normal, refreshed every PreProcess() call; used as the test
        # direction in the volume-conservation constraint below.
        self.pre_normal = GridFunction(V1)
        self.normal = GridFunction(V1)

        # Mixed FE space for the coupled (D, Y) system.
        fes = CompressCompound(V1*V2)
        self.A = BilinearForm(fes)
        self.F = LinearForm(fes)

        self.deform = self._solver.mesh.prev_deformation[-1]

        # Vertex-based (mass-lumped) quadrature rules, used to make P1
        # mass-matrix-like blocks diagonal (stabilizes the mixed system and
        # cheapens the curvature reconstruction / Lagrange-multiplier
        # integrals below).
        ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
        
        self.ds_lumped = ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = self.deform)
        ds_el_lumped = ds(element_boundary=True, intrules = { SEGM : ir_segm, TRIG: ir_trig })

        # Auxiliary "initial mean curvature" projection problem (mixed weak
        # form of the vector mean curvature kappa_mc = H*n, see
        # mean_curvature_boundary_bdf1_model.py). Used in PreProcess to
        # (re-)initialize the lagged curvature gfu_k_old/gfu_Y_old, either
        # once (iter==0) or every step (autoupdate=True).
        fes_mc = Compress(V2)
        kappa_mc, eta_mc = fes_mc.TnT()
        self.A_mc = BilinearForm(fes_mc, symmetric = True)
        self.A_mc += (kappa_mc*eta_mc)*self.ds_lumped
        self.A_mc.Assemble()
        self.invA_mc = self.A_mc.mat.Inverse(freedofs = fes_mc.FreeDofs())
        self.F_mc = LinearForm(fes_mc)
        self.F_mc += -InnerProduct(self.Ps, grad(eta_mc).Trace())*ds(deformation = self.deform)
        # if self.input_params["clamped_bnd"]:
        #     if self._solver.ngsmesh.dim == 2:
        #         gfBB = GridFunction(H1(self._solver.ngsmesh, order =1,\
        #                 definedon=self.domain))
        #         gfBB.Set(1, definedon=self._solver.ngsmesh.BBoundaries(self.clamped_bnd))
        #         self.F_mc += InnerProduct(gfBB*nE, eta_mc)*ds_el_lumped
        #     elif self._solver.ngsmesh.dim == 3:
        #         gfBB = GridFunction(FacetSurface(self._solver.ngsmesh, order=0, definedon = self.domain))
        #         gfBB.Set(1, definedon=self._solver.ngsmesh.BBoundaries(self.clamped_bnd))
        #         self.F_mc += InnerProduct(nE, eta_mc)*gfBB*ds_el_lumped
        # Natural boundary term prescribing the co-normal on the clamped
        # BBoundary (unreachable here, since clamped_bnd is rejected above;
        # kept for parity with the sibling models).
        if self.input_params["clamped_bnd"]:
            if self._solver.ngsmesh.dim == 2:
                gfBB = GridFunction(H1(self._solver.ngsmesh, order =1,\
                        definedon=self.domain))
                gfBB.Set(1, definedon=self._solver.ngsmesh.BBoundaries(self.input_params["clamped_bnd"]))
                self.F_mc += InnerProduct(gfBB*self.input_params["clamped_conormal"], eta_mc)*ds_el_lumped
            elif self._solver.ngsmesh.dim == 3:
                gfBB = GridFunction(FacetSurface(self._solver.ngsmesh, order=0, definedon = self.domain))
                gfBB.Set(1, definedon=self._solver.ngsmesh.BBoundaries(self.input_params["clamped_bnd"]))
                self.F_mc += InnerProduct(self.input_params["clamped_conormal"], eta_mc)*gfBB*ds_el_lumped
        self.F_mc.Assemble()

        (trial_D, trial_Y), (test_D, test_Y) = fes.TnT()
        self.gfu = GridFunction(fes)
        self.gfu_old = GridFunction(fes)
        # D: surface displacement/velocity unknown. Y = kappa_elastic*(H -
        # H0*n): auxiliary curvature-flux unknown. gfu_k: mean curvature H*n
        # reconstructed from Y after each solve; gfu_k_old: its lagged value.
        self.gfu_D, self.gfu_Y = self.gfu.components
        self.gfu_D_old, self.gfu_Y_old = self.gfu_old.components
        self.gfu_k = GridFunction(V2)
        self.gfu_k_old = GridFunction(V2)

        self.output_fields["displacement"] = OutputField(self.gfu_D, "displacement", BND)
        self.output_fields["mean_curvature"] = OutputField(self.gfu_k, "mean_curvature", BND)

        self.k0_gfu = GridFunction(V3)
        rhs_gfu = GridFunction(V2)
        kappa_gfu = GridFunction(V3)
        self.input_fields["spontaneous_curvature"] = InputField(self.k0_gfu, CF(0), "spontaneous_curvature", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["rhs"] = InputField(rhs_gfu, CF((0,)*self._solver.ngsmesh.dim), "rhs", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["elasticity_modulus"] = InputField(kappa_gfu, CF(1), "elasticity_modulus", self._solver.ngsmesh.Boundaries('.*'))
        kappa_gfu.Set(1, definedon = self.domain)
        

        # Reference position (identity map) and (unused elsewhere in this
        # file) accumulator for the total displacement of the surface.
        self.X0 = GridFunction(V1)
        if self._solver.ngsmesh.dim == 2:
            self.X0.Set(CF((x,y)), dual = True, definedon=self.domain)
        elif self._solver.ngsmesh.dim == 3:
            self.X0.Set(CF((x, y, z)), dual = True, definedon=self.domain)
        self.displacement_tot = GridFunction(V1)

        # BDF1 mass term for D plus the (lumped) mass term for Y, from
        # inverting Y = kappa_elastic*(H - H0*n).
        self.A += (1/self._solver.time.dt*trial_D*test_D + 1/kappa_gfu*InnerProduct(trial_Y, test_Y))*self.ds_lumped
        # Mixed-form coupling between D and Y: weak (first-order) form of the
        # surface Laplace-Beltrami operator, split to avoid second
        # derivatives on either test/trial function.
        self.A += (-InnerProduct(grad(trial_Y).Trace(), grad(test_D).Trace()) + InnerProduct(grad(trial_D).Trace(), grad(test_Y).Trace()))*ds(deformation = self.deform)
        
        # Same weak Laplace-Beltrami identity as in F_mc above, driving Y.
        self.F += (-InnerProduct(self.Ps, grad(test_Y).Trace()))*ds(deformation = self.deform)
        # Spontaneous-curvature contribution to Y = kappa*(H - H0*n).
        self.F += (-1*self.k0_gfu*InnerProduct(self.ns, test_Y))*self.ds_lumped
        self.F += (InnerProduct(rhs_gfu, test_D))*self.ds_lumped

        # Explicit/semi-implicit linearization of the nonlinear Willmore
        # (Helfrich) force about the lagged curvature state
        # (gfu_Y_old, gfu_k_old), keeping the D/Y block linear each step.
        self.F += (InnerProduct(Trace(grad(self.gfu_Y_old).Trace()),Trace(grad(test_D).Trace())))*ds(deformation = self.deform)
        self.F += (-2*InnerProduct(grad(self.gfu_Y_old).Trace().trans, self.D_s(test_D, self.Ps)*self.Ps.trans))*ds(deformation = self.deform)
        self.F += (-kappa_gfu*InnerProduct(self.k0_gfu*self.gfu_k_old, grad(test_D).Trace().trans*self.ns))*self.ds_lumped
        self.F += (-0.5*InnerProduct(kappa_gfu*(Norm(self.gfu_k_old - self.k0_gfu*self.ns)**2)*self.Ps,grad(test_D).Trace()))*self.ds_lumped
        self.F += (InnerProduct(InnerProduct(self.gfu_Y_old, self.gfu_k_old)*self.Ps,grad(test_D).Trace()))*self.ds_lumped

        if self.input_params["clamped_bnd"]:
            self.F += (InnerProduct(self.input_params["clamped_conormal"], test_Y)*gfBB)*ds_el_lumped
        
        # if self.input_params["clamped_bnd"]:
        #     self.F += (InnerProduct(nE, test_Y)*gfBB)*ds_el_lumped

        # Volume constraint (as in the VP variant): omega_h multiplies the
        # fixed surface normal self.normal, so it only enters F.
        self.F += (self.omega_h*InnerProduct(self.normal, test_D))*self.ds_lumped
        # Area constraint (as in the AP variant): lambda_h multiplies the
        # trial function's tangential gradient, so it enters both A and F.
        # Both multipliers are re-evaluated jointly (2x2 solve) and A/F
        # re-assembled every fixed-point iteration in Solve().
        self.A += (self.lambda_h*InnerProduct(grad(trial_D).Trace(), grad(test_D).Trace()))*ds(deformation = self.deform)
        self.F += -1*(self.lambda_h*InnerProduct(self.Ps, grad(test_D).Trace()))*ds(deformation = self.deform)

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = fes.FreeDofs())

    def D_s(self, chi, Ps):
        """Symmetric tangential derivative (surface rate-of-strain-like
        operator): the tangential projection Ps of the symmetrized tangential
        gradient of `chi`."""
        sym = 0.5*Ps*(grad(chi).Trace()+grad(chi).Trace().trans)*Ps
        return sym

    def PreProcess(self):
        """Snapshot the previous mixed solution and refresh the (unit)
        surface normal used by the volume constraint, then (re-)initialize
        the lagged curvature `gfu_k_old`/`gfu_Y_old` (also seeding
        `gfu_k`/`gfu_Y`) by solving the auxiliary projection problem
        (A_mc/F_mc) — always on the first time step, and on every step if
        `autoupdate` is set."""

        self.gfu_old.vec.data = self.gfu.vec.data
        self.pre_normal.Set(self.ns, dual = True, definedon = self.domain)
        self.normal.Set(Normalize(self.pre_normal), dual = True, definedon = self.domain)

        if self.input_params['autoupdate'] or self._solver.time.iter == 0:
            self.A_mc.Assemble()
            self.invA_mc.Update()
            self.F_mc.Assemble()
            self.gfu_k_old.vec.data = self.invA_mc*self.F_mc.vec
            self.gfu_Y_old.Set(self.input_fields["elasticity_modulus"].gfu*(self.gfu_k_old - self.input_fields["spontaneous_curvature"].gfu*self.ns), dual = True, definedon = self.domain)
            self.gfu_k.vec.data = self.gfu_k_old.vec.data
            self.gfu_Y.vec.data = self.gfu_Y_old.vec.data
        else:
            self.gfu_k_old.vec.data = self.gfu_k.vec.data
            self.gfu_Y_old.vec.data = self.gfu_Y.vec.data

    def Solve(self):
        """Refresh input fields, then run a fixed-point iteration that
        alternates jointly re-evaluating both Lagrange multipliers
        (`lambda_h` for area, `omega_h` for volume) by solving the coupled
        2x2 linear system implied by their two constraint equations, and
        re-solving the mixed (D, Y) system with the updated multipliers,
        until both stop changing (or raise after `maxiter` iterations)."""

        self.update_input_fields()
        
        self.gfu_D.vec.data[:]=0

        old_lambda = 0
        new_lambda = 1e5
        old_omega = 0
        new_omega = 1e5
        iter = 0

        maxiter = 20
        
        while abs(new_lambda-old_lambda) + abs(new_omega-old_omega)>1e-8 and iter<maxiter:

            iter += 1
            old_omega = self.omega_h.vec.data[0]
            old_lambda = self.lambda_h.vec.data[0]

            # F0: the same nonlinear Willmore force expression as in F above,
            # evaluated at the current iterate and tested against gfu_k — the
            # area-constraint equation (as in the AP variant's numerator).
            F0 = Integrate(
                    -InnerProduct(grad(self.gfu_Y).Trace(), grad(self.gfu_k).Trace())*ds(deformation = self.deform)
                    -1*InnerProduct(self.input_fields['rhs'].gfu, self.gfu_k)*self.ds_lumped
                    -1*InnerProduct(Trace(grad(self.gfu_Y).Trace()),Trace(grad(self.gfu_k).Trace()))*ds(deformation = self.deform)
                    +2*InnerProduct(grad(self.gfu_Y).Trace().trans, self.D_s(self.gfu_k, self.Ps)*self.Ps.trans)*ds(deformation = self.deform)
                    +self.input_fields["elasticity_modulus"].gfu*InnerProduct(self.k0_gfu*self.gfu_k, grad(self.gfu_k).Trace().trans*self.ns)*self.ds_lumped
                    +0.5*InnerProduct(self.input_fields["elasticity_modulus"].gfu*(Norm(self.gfu_k - self.k0_gfu*self.ns)**2)*self.Ps,grad(self.gfu_k).Trace())*self.ds_lumped
                    -InnerProduct(InnerProduct(self.gfu_Y, self.gfu_k)*self.Ps,grad(self.gfu_k).Trace())*self.ds_lumped
                , self._solver.ngsmesh)

            # F1: the same nonlinear Willmore force expression, but tested
            # against the surface normal self.normal instead of gfu_k — the
            # volume-constraint equation (as in the VP variant's numerator).
            F1 = Integrate(
                    -InnerProduct(grad(self.gfu_Y).Trace(), grad(self.normal).Trace())*ds(deformation = self.deform)
                    -1*InnerProduct(self.input_fields['rhs'].gfu, self.normal)*self.ds_lumped
                    -1*InnerProduct(Trace(grad(self.gfu_Y).Trace()),Trace(grad(self.normal).Trace()))*ds(deformation = self.deform)
                    +2*InnerProduct(grad(self.gfu_Y).Trace().trans, self.D_s(self.normal, self.Ps)*self.Ps.trans)*ds(deformation = self.deform)
                    +self.input_fields["elasticity_modulus"].gfu*InnerProduct(self.k0_gfu*self.gfu_k, grad(self.normal).Trace().trans*self.ns)*self.ds_lumped
                    +0.5*InnerProduct(self.input_fields["elasticity_modulus"].gfu*(Norm(self.gfu_k - self.k0_gfu*self.ns)**2)*self.Ps,grad(self.normal).Trace())*self.ds_lumped
                    -InnerProduct(InnerProduct(self.gfu_Y, self.gfu_k)*self.Ps,grad(self.normal).Trace())*self.ds_lumped
                , self._solver.ngsmesh)

            # 2x2 (symmetric) mass-like matrix from the "denominators" of the
            # two Rayleigh-quotient constraints (AP uses (gfu_k, gfu_k), VP
            # uses (normal, normal)); the off-diagonal A01=A10 couples the
            # two constraints since gfu_k and normal are not orthogonal in
            # general. Solving A @ [lambda_h, omega_h] = [F0, F1] gives the
            # joint (linearized) update for both multipliers simultaneously,
            # replacing the two independent scalar updates used by the AP/VP
            # variants individually.
            A00 = Integrate(InnerProduct(self.gfu_k, self.gfu_k)*self.ds_lumped, self._solver.ngsmesh)
            A01 = Integrate(InnerProduct(self.normal, self.gfu_k)*self.ds_lumped, self._solver.ngsmesh)
            A10 = A01
            A11 = Integrate(InnerProduct(self.normal, self.normal)*self.ds_lumped, self._solver.ngsmesh)

            A = np.array([[A00, A01], [A10, A11]])
            F = np.array([F0, F1])
            X = np.linalg.solve(A, F)

            self.lambda_h.vec.data[0] = X[0]
            new_lambda = X[0]
            self.omega_h.vec.data[0] = X[1]
            new_omega = X[1]

            # Re-assemble A/F with the updated multipliers and re-solve.
            self.A.Assemble()
            self.invA.Update()
            self.F.Assemble()

            self.gfu.vec.data = self.invA*self.F.vec
            self.gfu_k.Set(1/self.input_fields["elasticity_modulus"].gfu*self.gfu_Y + self.input_fields["spontaneous_curvature"].gfu*self.ns, dual = True, definedon = self.domain)

        if iter == maxiter:
            raise Exception('Convergence not achieved for Volume and Surface area preserving Willmore flow')

    def PostProcess(self):
        """No auxiliary state to advance for this model."""

        pass

    @property
    def displacement(self):
        """The current surface displacement/velocity field `D`."""
        return self.gfu_D
    
    @displacement.setter
    def phase(self, cf):
        """Note: due to the setter's method name not matching the property
        name, this does not actually override `displacement` (kept as-is for
        parity with the sibling model it was copied from)."""
        self.gfu_D.Set(cf, definedon = self.domain)

    @property
    def mean_curvature(self):
        """The current (vector) mean curvature H*n, reconstructed from Y."""
        return self.gfu_k
    
    @mean_curvature.setter
    def potential(self, cf):
        """Note: due to the setter's method name not matching the property
        name, this does not actually override `mean_curvature` (kept as-is
        for parity with the sibling model it was copied from)."""
        self.gfu_k.Set(cf, definedon = self.domain)

    @property
    def energy(self):
        """Discrete Helfrich bending energy
        0.5*kappa_elastic*integral((H - H0*n)^2) evaluated at the current
        mean curvature."""
        energy = Integrate(0.5*self.input_fields['elasticity_modulus'].gfu*Norm(self.mean_curvature\
                            -self.input_fields['spontaneous_curvature'].gfu*specialcf.normal(self._solver.ngsmesh.dim))**2, 
                            self._solver.ngsmesh, VOL_or_BND = BND)
        return energy