# Surface Willmore (Helfrich bending) flow, first-order BDF1 in time,
# "Inexact" (approximately area-preserving) variant. Solves the same mixed
# (D, Y) system as the baseline model (willmore_boundary_bdf1_model.py; see
# that file for the physics of the split L2-gradient flow of E = 0.5*
# kappa_elastic*integral((H-H0)^2)), extended with a third unknown field
# `lam` (a scalar P1 field, not a single global Lagrange multiplier as in the
# AP/VP/APVP variants) that reacts against the mixed system to approximately
# enforce conservation of surface area. Unlike the AP/VP/APVP variants, the
# nonlinear coefficients coupling `lam` back into the D/Y system are
# evaluated at the *lagged* (previous-step) curvature state rather than the
# current unknowns — see the "exact" (nonlinear, commented-out) formulation
# left in place for reference just above the linearized terms actually used.
# This "inexact" linearization keeps the whole three-field (D, Y, lam) system
# linear, so it is assembled and solved once per time step with no
# fixed-point/Newton sub-loop, unlike the AP/VP/APVP variants — trading exact
# area conservation for a much cheaper step. A runtime guard in Solve() raises
# if the discrete area drifts more than 10% from its initial value. See
# docs/pde_models.md.

import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.solver import Solver
from cosmos.core.field import InputField, OutputField
from ngsolve.webgui import Draw

class WillmoreBoundaryInexBDF1Model(BasePDEModel):
    """"Inexact" area-preserving BDF1 mixed FE solver for Willmore (Helfrich
    bending) flow of a boundary (surface) mesh.

    Extends the baseline mixed (D, Y) formulation (WillmoreBoundaryBDF1Model)
    with a third scalar field unknown `lam`, discretized as its own P1 field
    (with a Laplace-Beltrami-type regularization term) rather than a single
    global Lagrange multiplier. `lam` reacts against the D equation weighted
    by the lagged mean curvature, approximately enforcing area conservation
    without requiring the fixed-point/Newton sub-iteration used by
    WillmoreBoundaryAPBDF1Model: all nonlinear coefficients in the `lam`
    equation are evaluated at the previous step's (lagged) curvature, so the
    combined (D, Y, lam) system stays linear and is solved once per step.
    Solve() raises if the resulting drift in total surface area exceeds 10%
    of its initial value.
    """

    def __init__(self, solver:Solver,
                 model_order:int,
                 domain:str = '.*',
                 name:str = 'WillmoreBoundaryInexBDF1Model',
                 input_params = {}):
        """
        Args:
            solver: The Solver this model attaches to.
            model_order: Execution order among the solver's attached models.
            domain: Boundary region name(s) the equation is posed on (default: all).
            name: Unique name for this model instance within the solver.
            input_params: Overrides for the default scalar options (see below).

        Recognized input_params keys:
            clamped_bnd: BBoundary (edge) region name on which the co-normal
                is pinned to `clamped_conormal` (open/clamped membrane); ''
                (default) means no clamped edge.
            clamped_conormal: Prescribed co-normal vector on `clamped_bnd`.
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
        if (self.input_params["clamped_bnd"]!='') and \
            (self.input_params["clamped_bnd"] not in self._solver.mesh.bbnd_markers):
            raise ValueError('Clamped boundary conditions are imposed on non-existing BBoundary')

        h = self.cfg.h
        # Surface differential geometry quantities: tE/nE are the (edge)
        # tangent and co-normal of the boundary of the surface (used only when
        # clamped_bnd is set), ns is the surface unit normal, and Ps is the
        # tangential projector (I - n (x) n) used throughout to restrict
        # gradients to the surface tangent plane.
        tE = specialcf.tangential(self._solver.ngsmesh.dim)
        self.ns = specialcf.normal(self._solver.ngsmesh.dim)
        Ps = Id(self._solver.ngsmesh.dim) - OuterProduct(self.ns, self.ns)
        if self._solver.ngsmesh.dim == 2:
            nE = tE
            tEc = CF((-self.ns[1], self.ns[0]))
        else:
            nE = Cross(self.ns, tE)

        # TODO: Check is this can actually be merged 
        # V1: displacement space for D. When clamped_bnd is set, D is pinned
        # to zero (Dirichlet) on that BBoundary edge; the prescribed
        # co-normal itself is imposed weakly, as a natural boundary term on
        # the Y equation (see F below).
        if self.input_params["clamped_bnd"]:
            V1 = VectorH1(self._solver.ngsmesh, order=1, definedon=self.domain,
                    dirichlet_bbnd = self._solver.ngsmesh.BBoundaries(self.input_params["clamped_bnd"]))
        else:
            V1 = VectorH1(self._solver.ngsmesh, order=1,
                        definedon=self.domain)
        # V2: space for the auxiliary variable Y (and the mean curvature
        # gfu_k). V3: scalar space, used both for the material coefficients
        # and (unlike the baseline model) for the area-multiplier field `lam`.
        V2 = VectorH1(self._solver.ngsmesh, order=1, definedon=self.domain)
        V3 = H1(self._solver.ngsmesh, order=1, definedon=self.domain)

        # Mixed FE space for the coupled (D, Y, lam) system — a third field
        # compared to the baseline model's (D, Y) space.
        fes = CompressCompound(V1*V2*V3)
        self.A = BilinearForm(fes)
        self.F = LinearForm(fes)

        deform = self._solver.mesh.prev_deformation[-1]

        # Vertex-based (mass-lumped) quadrature rules: using these in place of
        # exact integration on P1 mass-matrix-like terms makes those blocks
        # diagonal, which stabilizes the mixed system and cheapens the
        # curvature reconstruction below.
        ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
        
        ds_lumped = ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = deform)
        ds_el_lumped = ds(element_boundary=True, intrules = { SEGM : ir_segm, TRIG: ir_trig })

        # Auxiliary "initial mean curvature" projection problem: mixed weak
        # form of the vector mean curvature kappa_mc = H*n via
        #   (kappa_mc, eta_mc) = -(Ps : grad(eta_mc))   for all eta_mc in V2,
        # i.e. the identity H*n = -Delta_Gamma X recast without second
        # derivatives. Used in PreProcess to (re-)initialize gfu_k_old, either
        # once (iter==0) or every step (autoupdate=True); also solved once
        # immediately below to seed gfu_k_old before self.A is assembled
        # (needed here, unlike in the baseline model, because gfu_k_old
        # already appears as a coefficient in A — see the trial_lam terms).
        fes_mc = V2
        kappa_mc, eta_mc = fes_mc.TnT()
        self.A_mc = BilinearForm(fes_mc, symmetric = True)
        self.A_mc += (kappa_mc*eta_mc)*ds_lumped
        self.A_mc.Assemble()
        self.invA_mc = self.A_mc.mat.Inverse(freedofs = fes_mc.FreeDofs())
        self.F_mc = LinearForm(fes_mc)
        self.F_mc += -InnerProduct(Ps, grad(eta_mc).Trace())*ds(deformation = deform)
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
        # BBoundary (weighted by an indicator gfBB that is 1 on that
        # edge/facet-strip and 0 elsewhere), entering the mean-curvature
        # projection's right-hand side.
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

        (trial_D, trial_Y, trial_lam), (test_D, test_Y, test_lam) = fes.TnT()
        self.gfu = GridFunction(fes)
        self.gfu_old = GridFunction(fes)
        # D: surface displacement/velocity unknown. Y = kappa_elastic*(H -
        # H0*n): auxiliary curvature-flux unknown. lam: scalar area-multiplier
        # field (this variant's addition relative to the baseline model).
        # gfu_k: mean curvature H*n reconstructed from Y after each solve;
        # gfu_k_old: its lagged value, used both to linearize the D/Y forcing
        # (as in the baseline model) and, here, to linearize the lam terms.
        self.gfu_D, self.gfu_Y, self.gfu_lam = self.gfu.components
        self.gfu_D_old, self.gfu_Y_old, self.gfu_lam_old = self.gfu_old.components
        self.gfu_k = GridFunction(V2)
        self.gfu_k_old = GridFunction(V2)

        # Seed gfu_k_old/gfu_k here (rather than waiting for the first
        # PreProcess call) because gfu_k_old is used as a coefficient while
        # building self.A below, and self.A is assembled once, at the end of
        # this constructor.
        self.A_mc.Assemble()
        self.invA_mc.Update()
        self.F_mc.Assemble()
        self.gfu_k_old.vec.data = self.invA_mc*self.F_mc.vec
        self.gfu_k.vec.data = self.gfu_k_old.vec.data

        self.output_fields["displacement"] = OutputField(self.gfu_D, "displacement", BND)
        self.output_fields["mean_curvature"] = OutputField(self.gfu_k, "mean_curvature", BND)
        self.output_fields["multiplier"] = OutputField(self.gfu_lam, "mean_curvature", BND)

        k0_gfu = GridFunction(V3)
        rhs_gfu = GridFunction(V2)
        kappa_gfu = GridFunction(V3)
        self.input_fields["spontaneous_curvature"] = InputField(k0_gfu, CF(0), "spontaneous_curvature", self._solver.ngsmesh.Boundaries('.*'))
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

        # BDF1 mass term for D plus the (lumped) mass term coupling trial_Y to
        # test_Y, coming from inverting Y = kappa_elastic*(H - H0*n).
        self.A += (1/self._solver.time.dt*trial_D*test_D + 1/kappa_gfu*InnerProduct(trial_Y, test_Y))*ds_lumped
        # Mixed-form coupling between D and Y: this antisymmetric pairing of
        # tangential gradients is the weak (first-order) form of the surface
        # Laplace-Beltrami operator linking D to Y, split so neither
        # test/trial function needs second derivatives. Identical to the
        # baseline model.
        self.A += (-InnerProduct(grad(trial_Y).Trace(), grad(test_D).Trace()) + InnerProduct(grad(trial_D).Trace(), grad(test_Y).Trace()))*ds(deformation = deform)
        
        # Same weak Laplace-Beltrami identity as in F_mc above, now driving Y.
        self.F += (-InnerProduct(Ps, grad(test_Y).Trace()))*ds(deformation = deform)
        # Spontaneous-curvature contribution to Y = kappa*(H - H0*n).
        self.F += (-1*k0_gfu*InnerProduct(self.ns, test_Y))*ds_lumped
        self.F += (InnerProduct(rhs_gfu, test_D))*ds_lumped

        def D_s(chi, Ps):
            """Symmetric tangential derivative (surface rate-of-strain-like
            operator): the tangential projection Ps of the symmetrized
            tangential gradient of `chi`."""
            sym = 0.5*Ps*(grad(chi).Trace()+grad(chi).Trace().trans)*Ps
            return sym

        # Explicit/semi-implicit treatment of the nonlinear bending force,
        # identical to the baseline model: linearized about the lagged
        # curvature state (gfu_Y_old, gfu_k_old) rather than the new unknowns.
        self.F += (InnerProduct(Trace(grad(self.gfu_Y_old).Trace()),Trace(grad(test_D).Trace())))*ds(deformation = deform)
        self.F += (-2*InnerProduct(grad(self.gfu_Y_old).Trace().trans, D_s(test_D, Ps)*Ps.trans))*ds(deformation = deform)
        self.F += (-kappa_gfu*InnerProduct(k0_gfu*self.gfu_k_old, grad(test_D).Trace().trans*self.ns))*ds_lumped
        self.F += (-0.5*InnerProduct(kappa_gfu*(Norm(self.gfu_k_old - k0_gfu*self.ns)**2)*Ps,grad(test_D).Trace()))*ds_lumped
        self.F += (InnerProduct(InnerProduct(self.gfu_Y_old, self.gfu_k_old)*Ps,grad(test_D).Trace()))*ds_lumped

        if self.input_params["clamped_bnd"]:
            # Natural boundary term imposing the prescribed co-normal on the
            # clamped edge, entering the main system's Y equation.
            self.F += (InnerProduct(self.input_params["clamped_conormal"], test_Y)*gfBB)*ds_el_lumped
        
        # if self.input_params["clamped_bnd"]:
        #     self.F += (InnerProduct(nE, test_Y)*gfBB)*ds_el_lumped

        # --- Area-multiplier field `lam`: this is where this variant departs
        # from the baseline model. ---
        #
        # "Exact" formulation (commented out, kept for reference): this would
        # couple trial_lam/test_lam to the *current* unknowns self.gfu_k and
        # self.gfu_Y. Since those are themselves being solved for, using them
        # as coefficients here makes the system nonlinear/solution-dependent
        # — solving it correctly would require an outer Newton/fixed-point
        # loop in Solve(), like the AP/VP/APVP variants.
        # self.A += (-1*trial_lam*self.gfu_k*test_D)*ds_lumped
        # self.A += (InnerProduct(grad(trial_lam).Trace(), grad(test_lam).Trace()))*ds(deformation = deform)
        # self.A += (trial_lam*Norm(self.gfu_k)**2*test_lam)*ds_lumped
        # self.F += -InnerProduct(grad(self.gfu_Y).Trace(), grad(self.gfu_k).Trace())*test_lam*ds(deformation = deform) \
        #             -1*InnerProduct(self.input_fields['rhs'].gfu, self.gfu_k)*test_lam*ds_lumped \
        #             -1*InnerProduct(Trace(grad(self.gfu_Y).Trace()),Trace(grad(self.gfu_k).Trace()))*test_lam*ds(deformation = deform) \
        #             +2*InnerProduct(grad(self.gfu_Y).Trace().trans, D_s(self.gfu_k, Ps)*Ps.trans)*test_lam*ds(deformation = deform) \
        #             +self.input_fields["elasticity_modulus"].gfu*InnerProduct(k0_gfu*self.gfu_k, grad(self.gfu_k).Trace().trans*self.ns)*test_lam*ds_lumped \
        #             +0.5*InnerProduct(self.input_fields["elasticity_modulus"].gfu*(Norm(self.gfu_k - k0_gfu*self.ns)**2)*Ps,grad(self.gfu_k).Trace())*test_lam*ds_lumped \
        #             -InnerProduct(InnerProduct(self.gfu_Y, self.gfu_k)*Ps,grad(self.gfu_k).Trace())*test_lam*ds_lumped \
        
        # "Inexact" formulation actually used: identical in structure to the
        # commented-out block above, but every occurrence of the current
        # unknowns (self.gfu_k, self.gfu_Y) is replaced by their *lagged*
        # (previous-step) counterparts (self.gfu_k_old, self.gfu_Y_old).
        # Since those are known GridFunctions (not trial functions), all
        # three added terms become genuinely linear/bilinear, so the whole
        # (D, Y, lam) system can be assembled once and solved directly, with
        # no outer iteration. This trades exact enforcement of the area
        # constraint (evaluated one step behind) for that reduced cost — the
        # `abs(self.area-self.area0)/self.area0>0.1` guard in Solve() exists
        # precisely because this approximation can drift.
        self.A += (-1*trial_lam*self.gfu_k_old*test_D)*ds_lumped
        # Laplace-Beltrami regularization of the multiplier field itself:
        # gives `lam` its own (elliptic) equation instead of being a single
        # unconstrained global DOF, smoothing it across the surface.
        self.A += (InnerProduct(grad(trial_lam).Trace(), grad(test_lam).Trace()))*ds(deformation = deform)
        self.A += (trial_lam*Norm(self.gfu_k_old)**2*test_lam)*ds_lumped
        # Right-hand side for the lam equation: the same nonlinear Willmore
        # force expression used to drive D above, now evaluated at the lagged
        # state and tested against test_lam — this is the discretized area
        # constraint that `lam` is being solved to (approximately) satisfy.
        self.F += -InnerProduct(grad(self.gfu_Y_old).Trace(), grad(self.gfu_k_old).Trace())*test_lam*ds(deformation = deform) \
                    -1*InnerProduct(self.input_fields['rhs'].gfu, self.gfu_k_old)*test_lam*ds_lumped \
                    -1*InnerProduct(Trace(grad(self.gfu_Y_old).Trace()),Trace(grad(self.gfu_k_old).Trace()))*test_lam*ds(deformation = deform) \
                    +2*InnerProduct(grad(self.gfu_Y_old).Trace().trans, D_s(self.gfu_k_old, Ps)*Ps.trans)*test_lam*ds(deformation = deform) \
                    +self.input_fields["elasticity_modulus"].gfu*InnerProduct(k0_gfu*self.gfu_k_old, grad(self.gfu_k_old).Trace().trans*self.ns)*test_lam*ds_lumped \
                    +0.5*InnerProduct(self.input_fields["elasticity_modulus"].gfu*(Norm(self.gfu_k_old - k0_gfu*self.ns)**2)*Ps,grad(self.gfu_k_old).Trace())*test_lam*ds_lumped \
                    -InnerProduct(InnerProduct(self.gfu_Y_old, self.gfu_k_old)*Ps,grad(self.gfu_k_old).Trace())*test_lam*ds_lumped \
        
        # Unlike the baseline model, A now depends on gfu_k_old (via the
        # trial_lam terms above), which is why gfu_k_old was seeded earlier
        # in this constructor rather than left to the first PreProcess call.
        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = fes.FreeDofs())

        # Record the initial discrete surface area, used by Solve() to bound
        # how far the (only approximately enforced) area constraint drifts.
        self.area0 = self.area

    def PreProcess(self):
        """(Re-)initialize the lagged mean curvature `gfu_k_old`/`gfu_Y_old`
        by solving the auxiliary projection problem (A_mc/F_mc) — always on
        the first time step, and on every step if `autoupdate` is set;
        otherwise just snapshot the previous mean curvature and mixed
        solution (needed since, unlike the baseline model, `gfu_old` here
        includes the extra `lam` component)."""

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
            self.gfu_old.vec.data = self.gfu.vec.data

        # temporary testing
        # prova = GridFunction(H1(self._solver.ngsmesh))
        # prova.Set(Norm(self.gfu_k_old), definedon = self.domain)
        # Draw(prova)

    def Solve(self):
        """Refresh input fields, then assemble and solve the linear mixed
        (D, Y, lam) system once (no fixed-point sub-loop, since all
        coefficients were linearized about the lagged state when the forms
        were built — contrast with WillmoreBoundaryAPBDF1Model.Solve()),
        reconstruct the mean curvature from Y, and raise if the resulting
        surface area has drifted more than 10% from its initial value."""

        self.update_input_fields()

        # Commented-out alternative: a fixed-point loop re-solving the linear
        # system until gfu_lam stabilizes. Not needed by the "inexact"
        # formulation actually used above (A/F depend only on the lagged
        # state, not on the current solution), but would be the natural way
        # to iterate towards the "exact" (nonlinear) formulation instead.
        # old = self.gfu_lam.vec.Copy()
        # old.data[:] = 1e5
        # iter = 0
        # maxiter = 10

        # while Norm(old-self.gfu_lam.vec)>1e-8 and iter<maxiter:

        #     iter += 1
        #     old = self.gfu_lam.vec.Copy()

        #     self.A.Assemble()
        #     self.invA.Update()
        #     self.F.Assemble()

        #     self.gfu.vec.data = self.invA*self.F.vec
        #     self.gfu_k.Set(1/self.input_fields["elasticity_modulus"].gfu*self.gfu_Y + self.input_fields["spontaneous_curvature"].gfu*self.ns, dual = True, definedon = self.domain)
        
        # print(iter)
        # if iter == maxiter:
        #     raise Exception('Convergence not achieved for inextensible area Willmore flow')

        self.A.Assemble()
        self.invA.Update()
        self.F.Assemble()

        self.gfu.vec.data = self.invA*self.F.vec
        self.gfu_k.Set(1/self.input_fields["elasticity_modulus"].gfu*self.gfu_Y + self.input_fields["spontaneous_curvature"].gfu*self.ns, dual = True, definedon = self.domain)

        # Safety guard on the approximately-enforced area constraint: raise
        # if the discrete surface area has drifted too far from its value at
        # construction time (self.area0), since this variant only
        # approximately conserves area (see the module/class docstrings).
        if abs(self.area-self.area0)/self.area0>0.1:
            raise Exception('Area of the simulation has exceeded accepted limit')

    def PostProcess(self):
        """No auxiliary state to advance for this model."""

        pass

    @property
    def displacement(self):
        """The current surface displacement/velocity field `D`."""
        return self.gfu_D

    @property
    def mean_curvature(self):
        """The current (vector) mean curvature H*n, reconstructed from Y."""
        return self.gfu_k

    @property
    def multiplier(self):
        """The current area-multiplier field `lam`."""
        return self.gfu_lam
    
    @property
    def energy(self):
        """Discrete Helfrich bending energy
        0.5*kappa_elastic*integral((H - H0*n)^2) evaluated at the current
        mean curvature."""
        energy = Integrate(0.5*self.input_fields['elasticity_modulus'].gfu*Norm(self.mean_curvature\
                            -self.input_fields['spontaneous_curvature'].gfu*specialcf.normal(self._solver.ngsmesh.dim))**2, 
                            self._solver.ngsmesh, VOL_or_BND = BND)
        return energy
    
    @property
    def area(self):
        """Total discrete surface area of the current mesh configuration."""
        area = Integrate(1, self._solver.ngsmesh, VOL_or_BND = BND)
        return area