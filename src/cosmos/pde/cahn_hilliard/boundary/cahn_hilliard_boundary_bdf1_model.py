import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.solver import Solver
from cosmos.core.field import InputField, OutputField
from cosmos.core.utils import MandBP
from myngspy import *

class CahnHilliardBoundaryBDF1Model(BasePDEModel):

    def __init__(self, solver:Solver,
                 model_order:int,
                 domain:str = '.*',
                 name:str = 'CahnHilliardBoundaryBDF1Model',
                 input_params = {}):
        
        super().__init__()
        
        self.name = name
        self._solver = solver
        self.model_order = model_order

        if (domain in solver.mesh.bnd_markers) or domain == '.*':
            self.domain = solver.ngsmesh.Boundaries(domain)
        else:
            logger.error('The domain specified for the Model ', self.name, ' does not exist')

        solver._attach_model(self, self.order)

        # TODO: Implement boundary conditions
        # self.input_params["Neu_bnd"] = ''
        # self.input_params["Dir_bnd"] = ''
        self.input_params["periodic"] = False
        self.input_params["mass_preserving"] = False
        self.input_params["bounds"] = None
        self.input_params["fes_order"] = 1
        self.input_params["u0"] = CF(0)
        self.input_params["w0"] = CF(0)
        self.input_params["epsilon"] = 1
        self.input_params["theta1"] = 1
        self.input_params["theta2"] = 1

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
        deform_old = self._solver.mesh.prev_deformation[-2]

        # Creating GridFunctions for the Fields
        b_gfu = GridFunction(fes_vector)
        D_gfu = GridFunction(_fes)
        rhs_u_gfu = GridFunction(_fes)
        rhs_w_gfu = GridFunction(_fes)
        # TODO: Implement boundary conditions
        # gradu_gfu = GridFunction(fes_vector)
        # u_bnd_gfu = GridFunction(fes)
        self.input_fields["b"] = InputField(b_gfu, CF((0,)*solver.mesh.dim), "b", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["D"] = InputField(D_gfu, CF(1), "D", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["rhs_u"] = InputField(rhs_u_gfu, CF(0), "rhs_u", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["rhs_w"] = InputField(rhs_w_gfu, CF(0), "rhs_w", self._solver.ngsmesh.Boundaries('.*'))
        # TODO: Implement boundary conditions
        # self.input_fields["gradu_bnd"] = InputField(gradu_gfu, CF((0,)*solver.mesh.dim), "gradu_bnd", self._solver.ngsmesh.Boundaries('.*'))
        # self.input_fields["u_bnd"] = InputField(u_bnd_gfu, CF(0), "u_bnd", self._solver.ngsmesh.Boundaries('.*'))

        if self.input_params["mass_preserving"]:
            ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = {  SEGM : ir_segm, TRIG : ir_trig })
            self.Amp = BilinearForm(self.gfu_u.space, symmetric = True)
            u, v = self.gfu_u.space.TnT()
            self.Amp += u*v*ds_lumped
            self.Amp.Assemble()
            rows,cols,vals = self.Amp.mat.COO()
            self.weights = sp.csr_matrix((vals,(rows,cols))).diagonal()

        self.A += 1/self._solver.time.dt*trial_u*test_u*ds(deformation = deform)
        self.A += D_gfu*grad(trial_w).Trace()*grad(test_u).Trace()*ds(deformation = deform)
        self.A += trial_w*test_w*ds(deformation = deform)
        self.A += -self.input_params["epsilon"]*grad(trial_u).Trace()*grad(test_w).Trace()*ds(deformation = deform)
                
        # if self.input_params['Dir_bnd']:
        #     self.A += - d_gfu*InnerProduct(n, grad(trial))*test*ds(definedon = self.input_params['Dir_bnd'], skeleton=True, deformation = deform) \
        #         - d_gfu*InnerProduct(n, grad(test))*trial*ds(definedon = self.input_params['Dir_bnd'], skeleton=True, deformation = deform)\
        #         + d_gfu*alpha/h*trial*test*ds(definedon = self.input_params['Dir_bnd'], skeleton = True, deformation = deform)\

        # self.A += -b_gfu*grad(test) * trial*dx(deformation = deform)
        # self.A += IfPos(b_gfu*n, b_gfu*n*trial, CF(0))*test\
        #     *ds(deformation = deform)

        Precond_A = Preconditioner(self.A, "local") # TODO: multigrid preconditioner doesn't seem to work
        self.A.Assemble()
        self.invA = GMRESSolver(self.A.mat, Precond_A.mat, maxsteps = 1000)

        self.F += rhs_u_gfu*test_u*ds(deformation = deform)
        self.F += rhs_w_gfu*test_w*ds(deformation = deform)
        
        # if self.input_params['Dir_bnd']:
        #     self.F += d_gfu*alpha/h*u_bnd_gfu*test*ds(definedon = self.input_params['Dir_bnd'], skeleton = True, deformation = deform)\
        #         - d_gfu*InnerProduct(n, grad(test))*u_bnd_gfu*ds(definedon = self.input_params['Dir_bnd'], skeleton=True, deformation = deform)
        # if self.input_params['Neu_bnd']:
        #     self.F += d_gfu*gradu_gfu*n*test*ds(definedon = self.input_params['Neu_bnd'], deformation = deform)

        # self.F += -IfPos(b_gfu*n, CF(0), b_gfu*n*u_bnd_gfu)*test*ds(deformation = deform)

        self.F += self.input_params["theta1"]*(log(1+self.gfu_u_old) - log(1-self.gfu_u_old))*test_w*ds(deformation = deform_old)
        self.F += -1*self.input_params["theta2"]*self.gfu_u_old*test_w*ds(deformation = deform)
        self.F += 1/self._solver.time.dt*self.gfu_u_old*test_u*ds(deformation = deform_old)

    def PreProcess(self):
        
        self.gfu_old.vec.data = self.gfu.vec.data

        if self._solver.time.iter==0 and self.input_params["mass_preserving"]:
            gfu0_vec = self.gfu_u.vec.Copy().FV().NumPy()
            self.mass0 = np.sum(self.weights*gfu0_vec)

    def Solve(self):

        self.update_input_fields()

        self.A.Assemble()
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
        
        for field in self.output_fields.values():
            if field.vtk and self._solver.time.iter % field.sample_rate == 0:
                field.vtk.Do(time = self._solver.time.t.Get(), vb = field.domain)

    @property
    def phase(self):
        return self.gfu_u
    
    @phase.setter
    def phase(self, cf):
        self.gfu_u.Set(cf, definedon = self.domain)

    @property
    def potential(self):
        return self.gfu_w
    
    @potential.setter
    def potential(self, cf):
        self.gfu_w.Set(cf, definedon = self.domain)

        