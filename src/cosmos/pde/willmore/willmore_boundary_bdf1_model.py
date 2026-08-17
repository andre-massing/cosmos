# Surface Willmore (Helfrich bending) flow, first-order BDF1 in time, baseline
# (unconstrained) variant. Solves the L2-gradient flow of the Helfrich energy
#   E = 0.5 * kappa_elastic * integral_Gamma (H - H0)^2
# for an evolving surface Gamma, where H is the (vector) mean curvature and H0
# the spontaneous curvature. The resulting fourth-order (in space) evolution
# equation is split into a mixed, first-order-in-space system for the surface
# displacement `D` and an auxiliary variable
#   Y = kappa_elastic * (H - H0 * n),
# avoiding the need for C1 finite elements. See
# mean_curvature_boundary_bdf1_model.py for the smaller mixed mean-curvature
# sub-problem reused here (as A_mc/F_mc) to (re-)initialize the lagged
# curvature `gfu_k_old`. See docs/pde_models.md.

import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.solver import Solver
from cosmos.core.field import InputField, OutputField
from ngsolve.webgui import Draw

class WillmoreBoundaryBDF1Model(BasePDEModel):
    """Baseline (no area/volume constraint) BDF1 mixed FE solver for Willmore
    (Helfrich bending) flow of a boundary (surface) mesh.

    Couples the displacement unknown `D` with the auxiliary variable
    `Y = kappa_elastic*(H - H0*n)` in a single linear mixed system solved once
    per time step; the nonlinear geometric terms of the bending force are
    treated explicitly/semi-implicitly using the lagged mean curvature
    `gfu_k_old` (see PreProcess). Optionally supports a clamped
    (open-boundary) edge where the surface co-normal is pinned.
    """

    def __init__(self, solver:Solver,
                 model_order:int,
                 domain:str = '.*',
                 name:str = 'WillmoreBoundaryBDF1Model',
                 input_params = {}):
        """
        Args:
            solver: The Solver this model attaches to.
            model_order: Execution order among the solver's attached models.
            domain: Boundary region name(s) the equation is posed on (default: all).
            name: Unique name for this model instance within the solver.
            input_params: Overrides for the default scalar options (see below).

        Recognized input_params keys:
            clamped_bnd: BBoundary (edge) region name on which the co-normal is
                pinned to `clamped_conormal` (open/clamped membrane); '' (default)
                means no clamped edge (closed surface / natural free boundary).
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
        # to zero (Dirichlet) on that BBoundary edge (the boundary curve does
        # not move); the prescribed co-normal itself is imposed weakly, as a
        # natural boundary term on the Y equation (see F below).
        if self.input_params["clamped_bnd"]:
            V1 = VectorH1(self._solver.ngsmesh, order=1, definedon=self.domain,
                    dirichlet_bbnd = self._solver.ngsmesh.BBoundaries(self.input_params["clamped_bnd"]))
        else:
            V1 = VectorH1(self._solver.ngsmesh, order=1,
                        definedon=self.domain)
        # V2: space for the auxiliary variable Y (and for the mean curvature
        # gfu_k it is post-processed into). V3: scalar space for the material
        # coefficients (spontaneous curvature, elasticity modulus).
        V2 = VectorH1(self._solver.ngsmesh, order=1,definedon=self.domain)
        V3 = H1(self._solver.ngsmesh, order=1, definedon=self.domain)

        # Mixed FE space for the coupled (D, Y) system.
        fes = CompressCompound(V1*V2)
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
        # i.e. the identity H*n = -Delta_Gamma X (Laplace-Beltrami of the
        # position) recast without second derivatives. Used in PreProcess to
        # (re-)initialize gfu_k_old, either once (iter==0) or every step
        # (autoupdate=True).
        fes_mc = V2
        kappa_mc, eta_mc = fes_mc.TnT()
        self.A_mc = BilinearForm(fes_mc, symmetric = True)
        self.A_mc += (kappa_mc*eta_mc)*ds_lumped
        self.A_mc.Assemble()
        self.invA_mc = self.A_mc.mat.Inverse(freedofs = fes_mc.FreeDofs())
        self.F_mc = LinearForm(fes_mc)
        self.F_mc += -InnerProduct(Ps, grad(eta_mc).Trace())*ds(deformation = deform)
        # Superseded alternative clamped-boundary term (kept for reference,
        # left unmodified/untouched):
        # if self.input_params["clamped_bnd"]:
        #     if self._solver.ngsmesh.dim == 2:
        #         gfBB = GridFunction(H1(self._solver.ngsmesh, order =1,\
        #                 definedon=self.domain))
        #         gfBB.Set(1, definedon=self._solver.ngsmesh.BBoundaries(self.input_params["clamped_bnd"]))
        #         self.F_mc += InnerProduct(gfBB*nE, eta_mc)*ds_el_lumped
        #     elif self._solver.ngsmesh.dim == 3:
        #         gfBB = GridFunction(FacetSurface(self._solver.ngsmesh, order=0, definedon = self.domain))
        #         gfBB.Set(1, definedon=self._solver.ngsmesh.BBoundaries(self.input_params["clamped_bnd"]))
        #         self.F_mc += InnerProduct(nE, eta_mc)*gfBB*ds_el_lumped
        if self.input_params["clamped_bnd"]:
            # Natural boundary term prescribing the co-normal on the clamped
            # BBoundary (weighted by an indicator gfBB that is 1 on that
            # edge/facet-strip and 0 elsewhere), entering the mean-curvature
            # projection's right-hand side.
            if self._solver.ngsmesh.dim == 2:
                gfBB = GridFunction(H1(self._solver.ngsmesh, order=1, definedon=self.domain))
                gfBB.Set(1, definedon=self._solver.ngsmesh.BBoundaries(self.input_params["clamped_bnd"]))
                self.F_mc += InnerProduct(self.input_params["clamped_conormal"], eta_mc)*gfBB*ds_el_lumped
            elif self._solver.ngsmesh.dim == 3:
                gfBB = GridFunction(FacetSurface(self._solver.ngsmesh, order=0, definedon = self.domain))
                gfBB.Set(1, definedon=self._solver.ngsmesh.BBoundaries(self.input_params["clamped_bnd"]))
                self.F_mc += InnerProduct(self.input_params["clamped_conormal"], eta_mc)*gfBB*ds_el_lumped
        self.F_mc.Assemble()

        (trial_D, trial_Y), (test_D, test_Y) = fes.TnT()
        self.gfu = GridFunction(fes)
        self.gfu_old = GridFunction(fes)
        # Unpack the mixed unknowns: D is the surface displacement/velocity,
        # Y = kappa_elastic*(H - H0*n) is the auxiliary curvature-flux
        # variable. gfu_k stores the mean curvature H*n reconstructed from Y
        # after each solve (see Solve()); gfu_k_old is its lagged value used
        # to linearize the nonlinear terms below.
        self.gfu_D, self.gfu_Y = self.gfu.components
        self.gfu_D_old, self.gfu_Y_old = self.gfu_old.components
        self.gfu_k = GridFunction(V2)
        self.gfu_k_old = GridFunction(V2)

        self.output_fields["displacement"] = OutputField(self.gfu_D, "displacement", BND)
        self.output_fields["mean_curvature"] = OutputField(self.gfu_k, "mean_curvature", BND)

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
        # Laplace-Beltrami operator linking the geometric unknown D to the
        # curvature-flux unknown Y, split so neither test/trial function needs
        # second derivatives.
        self.A += (-InnerProduct(grad(trial_Y).Trace(), grad(test_D).Trace()) + InnerProduct(grad(trial_D).Trace(), grad(test_Y).Trace()))*ds(deformation = deform)
        
        # Same weak Laplace-Beltrami identity as in F_mc above (-Ps:grad(eta)),
        # now driving the Y equation instead of the standalone projection.
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

        # Explicit/semi-implicit treatment of the nonlinear bending force: the
        # following five terms are the linearization of the Willmore
        # (Helfrich) force about the lagged curvature state
        # (gfu_Y_old, gfu_k_old) rather than the new unknowns, which is what
        # keeps the overall system linear in (trial_D, trial_Y) each step.
        self.F += (InnerProduct(Trace(grad(self.gfu_Y_old).Trace()),Trace(grad(test_D).Trace())))*ds(deformation = deform)
        self.F += (-2*InnerProduct(grad(self.gfu_Y_old).Trace().trans, D_s(test_D, Ps)*Ps.trans))*ds(deformation = deform)
        self.F += (-kappa_gfu*InnerProduct(k0_gfu*self.gfu_k_old, grad(test_D).Trace().trans*self.ns))*ds_lumped
        self.F += (-0.5*InnerProduct(kappa_gfu*(Norm(self.gfu_k_old - k0_gfu*self.ns)**2)*Ps,grad(test_D).Trace()))*ds_lumped
        self.F += (InnerProduct(InnerProduct(self.gfu_Y_old, self.gfu_k_old)*Ps,grad(test_D).Trace()))*ds_lumped

        if self.input_params["clamped_bnd"]:
            # Natural boundary term imposing the prescribed co-normal on the
            # clamped edge, entering the main system's Y equation (analogous
            # to the F_mc clamped term above).
            self.F += (InnerProduct(self.input_params["clamped_conormal"], test_Y)*gfBB)*ds_el_lumped
        
        # if self.input_params["clamped_bnd"]:
        #     self.F += (InnerProduct(nE, test_Y)*gfBB)*ds_el_lumped

        # The mixed system matrix A does not depend on the solution, so it is
        # assembled and factorized once here; only F (and A in Solve(), since
        # the mesh deformation may have changed) needs to be refreshed and
        # re-solved every time step.
        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = fes.FreeDofs())

    def PreProcess(self):
        """Snapshot the previous mixed solution and mean curvature, then
        (re-)initialize the lagged mean curvature `gfu_k_old`/`gfu_Y_old` by
        solving the auxiliary projection problem (A_mc/F_mc) — always on the
        first time step, and on every step if `autoupdate` is set."""

        self.gfu_old.vec.data = self.gfu.vec.data
        self.gfu_k_old.vec.data = self.gfu_k.vec.data

        if self.input_params['autoupdate'] or self._solver.time.iter == 0:
            self.A_mc.Assemble()
            self.invA_mc.Update()
            self.F_mc.Assemble()
            self.gfu_k_old.vec.data = self.invA_mc*self.F_mc.vec
            # Rebuild gfu_Y_old consistently with the freshly-projected
            # curvature via the defining relation Y = kappa*(H - H0*n).
            self.gfu_Y_old.Set(self.input_fields["elasticity_modulus"].gfu*(self.gfu_k_old - self.input_fields["spontaneous_curvature"].gfu*self.ns), dual = True, definedon = self.domain)

            # temporary testing
            # prova = GridFunction(H1(self._solver.ngsmesh))
            # prova.Set(Norm(self.gfu_k_old), definedon = self.domain)
            # Draw(prova)

    def Solve(self):
        """Refresh input fields, re-assemble the (deformation-dependent)
        system, solve the linear mixed system for (D, Y), and reconstruct the
        mean curvature `gfu_k` from Y via the inverse relation
        H*n = Y/kappa_elastic + H0*n."""

        self.update_input_fields()

        self.A.Assemble()
        self.invA.Update()
        self.F.Assemble()

        self.gfu.vec.data = self.invA*self.F.vec
        self.gfu_k.Set(1/self.input_fields["elasticity_modulus"].gfu*self.gfu_Y + self.input_fields["spontaneous_curvature"].gfu*self.ns, dual = True, definedon = self.domain)

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
    def energy(self):
        """Discrete Helfrich bending energy
        0.5*kappa_elastic*integral((H - H0*n)^2) evaluated at the current
        mean curvature."""
        energy = Integrate(0.5*self.input_fields['elasticity_modulus'].gfu*Norm(self.mean_curvature\
                            -self.input_fields['spontaneous_curvature'].gfu*specialcf.normal(self._solver.ngsmesh.dim))**2, 
                            self._solver.ngsmesh, VOL_or_BND = BND)
        return energy