# Boundary (surface) advection-diffusion-reaction PDE model, second-order BDF
# (BDF2) time discretization. Solves the same equation as the volume model but
# intrinsically on a codimension-1 surface, using tangential (.Trace())
# gradients; Dirichlet/Neumann conditions are imposed along the surface's edge
# (codimension-2 "BBoundary") via a symmetric interior penalty (SIP)
# formulation and upwinding. See docs/pde_models.md.

import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.solver import Solver
from cosmos.core.field import InputField, OutputField
from cosmos.core.utils import MandBP
from ngsolve.webgui import Draw

class ADRBoundaryBDF2Model(BasePDEModel):
    """Second-order (BDF2) advection-diffusion-reaction solver posed intrinsically
    on a surface (codimension-1) domain, using tangential gradients and
    FacetSurface/BBoundary representations of the boundary conditions along the
    surface's edge. Requires two previous time levels: the first step falls back
    to a BDF1 (implicit Euler) update, and the true BDF2 coefficients are
    switched on from the second step onward."""

    def __init__(self, solver:Solver,
                 model_order:int,
                 domain:str = '.*',
                 name:str = 'ADRBoundaryBDF2Model',
                 input_params = {}):
        """
        Args:
            solver: The Solver this model attaches to.
            model_order: Execution order among the solver's attached models.
            domain: Boundary region name(s) the equation is posed on (default: all).
            name: Unique name for this model instance within the solver.
            input_params: Overrides for the default scalar options (see below).

        Recognized input_params keys: `Neu_bnd`/`Dir_bnd` (Neumann/Dirichlet edge
        region names, i.e. codimension-2 BBoundaries of the surface), `periodic`,
        `mass_preserving`, `bounds` (`[low, high]`), `fes_order`, `u0` (initial
        condition).
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

        self.input_params["Neu_bnd"] = ''
        self.input_params["Dir_bnd"] = ''
        self.input_params["periodic"] = False
        self.input_params["mass_preserving"] = False
        self.input_params["bounds"] = None
        self.input_params["fes_order"] = 1
        self.input_params["u0"] = CF(0)

        self.set_input_params(input_params)
        
        if self.input_params["periodic"]:
            fes = Compress(Periodic(H1(self._solver.ngsmesh, order = self.input_params["fes_order"], 
                                            definedon = self.domain)))
            fes_vector = Compress(Periodic(VectorH1(self._solver.ngsmesh, order = self.input_params["fes_order"], 
                                            definedon = self.domain)))
        else:
            fes = Compress(H1(self._solver.ngsmesh, order = self.input_params["fes_order"], 
                                   definedon = self.domain))
            fes_vector = Compress(VectorH1(self._solver.ngsmesh, order = self.input_params["fes_order"], 
                                   definedon = self.domain))
            
        n = specialcf.normal(self._solver.mesh.dim)
        Ps = Id(self._solver.ngsmesh.dim) - OuterProduct(n, n)
        h = self.cfg.h
        alpha = 5 * self.input_params["fes_order"] * (self.input_params["fes_order"]+1)
        tE = specialcf.tangential(self._solver.ngsmesh.dim)
        # In 2D the surface's edge is a set of points, and the outward co-normal
        # nE coincides with the surface's own tangent; in 3D the edge is a
        # genuine 1D facet, so edge (BBoundary) indicator data lives on a
        # FacetSurface space and nE = n x tE is the outward co-normal.
        if self._solver.ngsmesh.dim == 2:
            facet_space = fes
            nE = specialcf.tangential(self._solver.ngsmesh.dim)
        else:
            facet_space = FacetSurface(self._solver.ngsmesh, order = 0)
            nE = Cross(n, tE)
          
        trial, test = fes.TnT()
        self.A = BilinearForm(fes)
        self.F = LinearForm(fes)
        
        self.gfu = GridFunction(fes)
        self.gfu_old = GridFunction(fes)
        self.gfu_oldold = GridFunction(fes)
        self.gfu.Set(self.input_params['u0'], definedon = self.domain, dual = True)
        self.output_fields["sol"] = OutputField(self.gfu, "sol", BND)

        deform = self._solver.mesh.curr_deformation
        deform_old = self._solver.mesh.prev_deformation[-1]
        deform_oldold = self._solver.mesh.prev_deformation[-2]

        # Creating GridFunctions for the Fields
        b_gfu = GridFunction(fes_vector)
        d_gfu = GridFunction(fes)
        c_gfu = GridFunction(fes)
        rhs_gfu = GridFunction(fes)
        gradu_gfu = GridFunction(fes_vector)
        u_bnd_gfu = GridFunction(fes)
        self.input_fields["b"] = InputField(b_gfu, CF((0,)*solver.mesh.dim), "b", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["d"] = InputField(d_gfu, CF(0), "d", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["c"] = InputField(c_gfu, CF(0), "c", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["rhs"] = InputField(rhs_gfu, CF(0), "rhs", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["gradu_bnd"] = InputField(gradu_gfu, CF((0,)*solver.mesh.dim), "gradu_bnd", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["u_bnd"] = InputField(u_bnd_gfu, CF(0), "u_bnd", self._solver.ngsmesh.Boundaries('.*'))

        if self.input_params["mass_preserving"]:
            # Mass-lumped quadrature rules (vertex-based, over segments/triangles)
            # used to build a diagonal mass matrix whose entries serve as
            # quadrature weights for MandBP.
            ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = {  SEGM : ir_segm, TRIG : ir_trig }, deformation = deform)
            self.Amp = BilinearForm(self.gfu.space, symmetric = True)
            u, v = self.gfu.space.TnT()
            self.Amp += u*v*ds_lumped
            self.Amp.Assemble()
            rows,cols,vals = self.Amp.mat.COO()
            self.weights = sp.csr_matrix((vals,(rows,cols))).diagonal()

        # Reaction and (tangential) diffusion terms.
        self.A += c_gfu*trial*test*ds(deformation = deform)
        self.A += d_gfu*grad(trial).Trace()*grad(test).Trace()*ds(deformation = deform)
                
        if self.input_params['Dir_bnd']:
            # dir_bnd_gfu is a 0/1 indicator on facet_space marking the edge
            # elements lying in the Dirichlet region, since
            # ds(element_boundary=True) integrates over the whole edge skeleton
            # and this term must vanish outside it.
            dir_bnd_gfu = GridFunction(facet_space)
            dir_bnd_gfu.Set(1, definedon = self._solver.ngsmesh.BBoundaries(self.input_params['Dir_bnd']))
            # Symmetric interior penalty (Nitsche-type) weak imposition of the
            # Dirichlet condition on the diffusive flux, evaluated along the edge.
            self.A += - dir_bnd_gfu*d_gfu*InnerProduct(nE, grad(trial).Trace())*test*ds(element_boundary=True, deformation = deform) \
                - dir_bnd_gfu*d_gfu*InnerProduct(nE, grad(test).Trace())*trial*ds(element_boundary=True, deformation = deform)\
                + dir_bnd_gfu*d_gfu*alpha/h*trial*test*ds(element_boundary=True, deformation = deform)

        # Advection term, integrated by parts using the tangential gradient.
        self.A += -b_gfu*grad(test).Trace() * trial*ds(deformation = deform)
        # bnd_gfu marks the edge elements lying on the prescribed (Dirichlet or
        # Neumann) boundary; the upwind flux term below only contributes there.
        bnd_gfu = GridFunction(facet_space)
        bnd_gfu.Set(1, definedon = self._solver.ngsmesh.BBoundaries(self.input_params['Dir_bnd']+'|'+self.input_params['Neu_bnd']))
        # Upwind flux on the outflow portion of the edge (IfPos selects b.nE > 0).
        self.A += bnd_gfu*IfPos(b_gfu*nE, b_gfu*nE*trial, CF(0))*test\
            *ds(element_boundary=True, deformation = deform)
        
        # BDF2 time derivative coefficient: starts at 1 (i.e. a BDF1 first step)
        # and is reset to 3/2 in PreProcess() once a second previous level exists.
        self.alpha0 = Parameter(1.0)
        self.A += self.alpha0/self._solver.time.dt*trial*test*ds(deformation = deform)

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = fes.FreeDofs())

        self.F += rhs_gfu*test*ds(deformation = deform)
        
        if self.input_params['Dir_bnd']:
            self.F += dir_bnd_gfu*d_gfu*alpha/h*u_bnd_gfu*test*ds(element_boundary=True, deformation = deform)\
                - dir_bnd_gfu*d_gfu*InnerProduct(nE, grad(test).Trace())*u_bnd_gfu*ds(element_boundary=True, deformation = deform)
        if self.input_params['Neu_bnd']:
            neu_bnd_gfu = GridFunction(facet_space)
            neu_bnd_gfu.Set(1, definedon = self._solver.ngsmesh.BBoundaries(self.input_params['Neu_bnd']))
            self.F += neu_bnd_gfu*d_gfu*gradu_gfu*nE*test*ds(element_boundary=True, deformation = deform)

        self.F += -bnd_gfu*IfPos(b_gfu*nE, CF(0), b_gfu*nE*u_bnd_gfu)*test*ds(element_boundary=True, deformation = deform)

        # BDF2 combination of the two previous time levels; alpha1/alpha2 start
        # as 1/0 (BDF1 first step) and are reset to 2/-0.5 in PreProcess() once a
        # second previous level ("oldold") is available.
        self.alpha1 = Parameter(1.0)
        self.F += self.alpha1/self._solver.time.dt*self.gfu_old*test*ds(deformation = deform_old)
        self.alpha2 = Parameter(0.0)
        self.F += self.alpha2/self._solver.time.dt*self.gfu_oldold*test*ds(deformation = deform_oldold)

    def PreProcess(self):
        """Shift the previous-time-level snapshots (old -> oldold, current -> old).
        On the first iteration, validate that mass/bounds preservation is only
        requested for fes_order == 1 (not yet implemented otherwise), and record
        the target mass if mass preservation is requested; from the second
        iteration onward, switch the BDF2 coefficients on (1.5, 2, -0.5) now that
        a second previous level is available."""
        
        self.gfu_oldold.vec.data = self.gfu_old.vec.data
        self.gfu_old.vec.data = self.gfu.vec.data

        if self._solver.time.iter == 0:

            if self.input_params["mass_preserving"] and self.input_params["fes_order"]>1:
                raise Exception('Mass preservation not yet implemented for fes_order>1')
            if self.input_params["bounds"] and self.input_params["fes_order"]>1:
                raise Exception('Bounds preservation not yet implemented for fes_order>1')
        
            if self.input_params["mass_preserving"]:
                gfu0_vec = self.gfu.vec.Copy().FV().NumPy()
                self.mass0 = np.sum(self.weights*gfu0_vec)

        if self._solver.time.iter == 1:
            self.alpha0.Set(1.5)
            self.alpha1.Set(2)
            self.alpha2.Set(-0.5)

    def Solve(self):
        """Refresh input fields, re-assemble and solve the linear system for this
        step, then optionally apply bound/mass-preservation post-processing."""
        
        self._solver.time.advance_tcoef()
        self._solver.mesh.advance_mesh()
        self.update_input_fields()

        self.A.Assemble()
        self.invA.Update()
        self.F.Assemble()

        self.gfu.vec.data = self.invA*self.F.vec
        
        if self.input_params["bounds"] and not self.input_params["mass_preserving"]:

            gfu_vec = self.gfu.vec.Copy().FV().NumPy()
            gfu_new = MandBP(gfu_vec, BP = self.input_params["bounds"])
            self.gfu.vec.data = gfu_new

        elif self.input_params["mass_preserving"]:

            if hasattr(self._solver.time, 'dt'):
                dt = self._solver.time.dt.Get()
            else:
                logger.error('A time-dependent simulation is needed to impose conservative mass')

            self.Amp.Assemble()
            rows,cols,vals = self.Amp.mat.COO()
            weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
            gfu_vec = self.gfu.vec.Copy().FV().NumPy()

            if self.input_params["bounds"]:
                BP = self.input_params["bounds"]
            else:
                BP = [-np.inf, np.inf]

            gfu_new = MandBP(gfu_vec, weights=weights, BP=BP,
                                MP=self.input_params["mass_preserving"], mass0=self.mass0, dt = dt)

            self.gfu.vec.data = gfu_new

        self._solver.time.reset_tcoef()
        self._solver.mesh.reset_mesh()

    def PostProcess(self):
        """No auxiliary state to advance for this model."""
        
        pass

    @property
    def sol(self):
        """The current solution GridFunction."""
        return self.gfu
    
    @sol.setter
    def sol(self, cf):
        """Overwrite the current solution by interpolating a coefficient."""
        self.gfu.Set(cf, definedon = self.domain)

        