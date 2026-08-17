# Volume (bulk-domain) advection-diffusion-reaction PDE model, first-order BDF
# (implicit Euler) time discretization. Solves:
#   du/dt + div(b u) + c u - div(d grad u) = rhs
# with Dirichlet/Neumann boundary conditions imposed via a symmetric interior
# penalty (SIP) formulation, and optional periodicity, bound- and
# mass-preservation. See docs/pde_models.md.

import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.solver import Solver
from cosmos.core.field import InputField, OutputField
from cosmos.core.utils import MandBP

class ADRVolumeBDF1Model(BasePDEModel):
    """First-order (BDF1/implicit Euler) advection-diffusion-reaction solver on a
    volume (bulk) domain."""

    def __init__(self, solver:Solver,
                 model_order:int,
                 domain:str = '.*',
                 name:str = 'ADRVolumeBDF1Model',
                 input_params = {}):
        """
        Args:
            solver: The Solver this model attaches to.
            model_order: Execution order among the solver's attached models.
            domain: Material region name(s) the equation is posed on (default: all).
            name: Unique name for this model instance within the solver.
            input_params: Overrides for the default scalar options (see below).

        Recognized input_params keys: `Neu_bnd`/`Dir_bnd` (Neumann/Dirichlet
        boundary region names), `periodic`, `mass_preserving`, `bounds`
        (`[low, high]`), `fes_order`, `u0` (initial condition).
        """

        super().__init__()

        self.VorB = VOL
        self.name = name
        self._solver = solver
        self.model_order = model_order

        if solver.mesh.is_bnd:
            logger.error('The mesh is only a surface, use ADRBoundaryBDF1Model')

        if (domain in solver.mesh.vol_markers) or domain == '.*':
            self.domain = solver.ngsmesh.Materials(domain)
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

        n = specialcf.normal(self._solver.mesh.dim)
        h = self.cfg.h
        # Symmetric interior penalty (SIP) parameter, scaled with polynomial order.
        alpha = 5 * self.input_params["fes_order"] * (self.input_params["fes_order"]+1)

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
          
        trial, test = fes.TnT()
        self.A = BilinearForm(fes)
        self.F = LinearForm(fes)
        
        self.gfu = GridFunction(fes)
        self.gfu_old = GridFunction(fes)
        self.gfu.Set(self.input_params['u0'], dual = True)
        self.output_fields["sol"] = OutputField(self.gfu, "sol", VOL)

        deform = self._solver.mesh.curr_deformation
        deform_old = self._solver.mesh.prev_deformation[-1]

        # GridFunctions backing the model's named InputFields (advection velocity
        # b, diffusivity d, reaction coefficient c, source rhs, Neumann/Dirichlet
        # boundary data).
        b_gfu = GridFunction(fes_vector)
        d_gfu = GridFunction(fes)
        c_gfu = GridFunction(fes)
        rhs_gfu = GridFunction(fes)
        gradu_gfu = GridFunction(fes_vector)
        u_bnd_gfu = GridFunction(fes)
        self.input_fields["b"] = InputField(b_gfu, CF((0,)*solver.mesh.dim), "b", self._solver.ngsmesh.Materials('.*'))
        self.input_fields["d"] = InputField(d_gfu, CF(0), "d", self._solver.ngsmesh.Materials('.*'))
        self.input_fields["c"] = InputField(c_gfu, CF(0), "c", self._solver.ngsmesh.Materials('.*'))
        self.input_fields["rhs"] = InputField(rhs_gfu, CF(0), "rhs", self._solver.ngsmesh.Materials('.*'))
        self.input_fields["gradu_bnd"] = InputField(gradu_gfu, CF((0,)*solver.mesh.dim), "gradu_bnd", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["u_bnd"] = InputField(u_bnd_gfu, CF(0), "u_bnd", self._solver.ngsmesh.Boundaries('.*'))

        if self.input_params["mass_preserving"]:
            # Mass-lumped quadrature rules (vertex-based) used to build a diagonal
            # mass matrix whose entries serve as quadrature weights for MandBP.
            ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ir_tet = IntegrationRule(points  = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], \
                                     weights = [1/24, 1/24, 1/24, 1/24])
            dx_lumped = dx(deformation = deform, intrules = { TRIG : ir_trig , TET : ir_tet})
            self.Amp = BilinearForm(self.gfu.space, symmetric = True)
            u, v = self.gfu.space.TnT()
            self.Amp += u*v*dx_lumped
            self.Amp.Assemble()
            rows,cols,vals = self.Amp.mat.COO()
            self.weights = sp.csr_matrix((vals,(rows,cols))).diagonal()

        # Reaction and diffusion terms.
        self.A += c_gfu*trial*test*dx(deformation = deform)
        self.A += d_gfu*grad(trial)*grad(test)*dx(deformation = deform)

        if self.input_params['Dir_bnd']:
            # Symmetric interior penalty (Nitsche-type) weak imposition of the
            # Dirichlet condition on the diffusive flux.
            self.A += - d_gfu*InnerProduct(n, grad(trial))*test*ds(definedon = self.input_params['Dir_bnd'], skeleton=True, deformation = deform) \
                - d_gfu*InnerProduct(n, grad(test))*trial*ds(definedon = self.input_params['Dir_bnd'], skeleton=True, deformation = deform)\
                + d_gfu*alpha/h*trial*test*ds(definedon = self.input_params['Dir_bnd'], skeleton = True, deformation = deform)\

        # Advection term, integrated by parts, plus upwind flux on the outflow
        # boundary (IfPos selects the outgoing-flux branch b.n > 0).
        self.A += -b_gfu*grad(test) * trial*dx(deformation = deform)
        self.A += IfPos(b_gfu*n, b_gfu*n*trial, CF(0))*test\
            *ds(deformation = deform)

        # Implicit-Euler (BDF1) time derivative term.
        self.A += 1/self._solver.time.dt*trial*test*dx(deformation = deform)

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = fes.FreeDofs())

        self.F += rhs_gfu*test*dx(deformation = deform)
        
        if self.input_params['Dir_bnd']:
            self.F += d_gfu*alpha/h*u_bnd_gfu*test*ds(definedon = self.input_params['Dir_bnd'], skeleton = True, deformation = deform)\
                - d_gfu*InnerProduct(n, grad(test))*u_bnd_gfu*ds(definedon = self.input_params['Dir_bnd'], skeleton=True, deformation = deform)
        if self.input_params['Neu_bnd']:
            self.F += d_gfu*gradu_gfu*n*test*ds(definedon = self.input_params['Neu_bnd'], deformation = deform)

        self.F += -IfPos(b_gfu*n, CF(0), b_gfu*n*u_bnd_gfu)*test*ds(deformation = deform)

        # BDF1 time derivative term evaluated on the previous configuration.
        self.F += 1/self._solver.time.dt*self.gfu_old*test*dx(deformation = deform_old)

    def PreProcess(self):
        """Snapshot the current solution as the previous time level; on the first
        iteration, if mass preservation is requested, record the target mass."""

        self.gfu_old.vec.data = self.gfu.vec.data

        if self._solver.time.iter==0 and self.input_params["mass_preserving"]:
            gfu0_vec = self.gfu.vec.Copy().FV().NumPy()
            self.mass0 = np.sum(self.weights*gfu0_vec)

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

        