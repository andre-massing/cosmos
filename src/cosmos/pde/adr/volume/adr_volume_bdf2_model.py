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

class ADRVolumeBDF2Model(BasePDEModel):

    def __init__(self, solver:Solver,
                 model_order:int,
                 domain:str = '.*',
                 name:str = 'ADRVolumeBDF2Model',
                 input_params = {}):
        
        super().__init__()
        
        self.name = name
        self._solver = solver
        self.model_order = model_order

        if solver.mesh.is_bnd:
            logger.error('The mesh is only a surface, use ADRBoundaryBDF2Model')

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
        self.gfu_oldold = GridFunction(fes)
        self.gfu.Set(self.input_params['u0'], dual = True)
        self.output_fields["sol"] = OutputField(self.gfu, "sol", VOL)

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
        self.input_fields["b"] = InputField(b_gfu, CF((0,)*solver.mesh.dim), "b", self._solver.ngsmesh.Materials('.*'))
        self.input_fields["d"] = InputField(d_gfu, CF(0), "d", self._solver.ngsmesh.Materials('.*'))
        self.input_fields["c"] = InputField(c_gfu, CF(0), "c", self._solver.ngsmesh.Materials('.*'))
        self.input_fields["rhs"] = InputField(rhs_gfu, CF(0), "rhs", self._solver.ngsmesh.Materials('.*'))
        self.input_fields["gradu_bnd"] = InputField(gradu_gfu, CF((0,)*solver.mesh.dim), "gradu_bnd", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["u_bnd"] = InputField(u_bnd_gfu, CF(0), "u_bnd", self._solver.ngsmesh.Boundaries('.*'))

        if self.input_params["mass_preserving"]:
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

        self.A += c_gfu*trial*test*dx(deformation = deform)
        self.A += d_gfu*grad(trial)*grad(test)*dx(deformation = deform)
                
        if self.input_params['Dir_bnd']:
            self.A += - d_gfu*InnerProduct(n, grad(trial))*test*ds(definedon = self.input_params['Dir_bnd'], skeleton=True, deformation = deform) \
                - d_gfu*InnerProduct(n, grad(test))*trial*ds(definedon = self.input_params['Dir_bnd'], skeleton=True, deformation = deform)\
                + d_gfu*alpha/h*trial*test*ds(definedon = self.input_params['Dir_bnd'], skeleton = True, deformation = deform)\

        self.A += -b_gfu*grad(test) * trial*dx(deformation = deform)
        self.A += IfPos(b_gfu*n, b_gfu*n*trial, CF(0))*test\
            *ds(deformation = deform)
        
        self.alpha0 = Parameter(1.0)
        self.A += self.alpha0/self._solver.time.dt*trial*test*dx(deformation = deform)

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = fes.FreeDofs())

        self.F += rhs_gfu*test*dx(deformation = deform)
        
        if self.input_params['Dir_bnd']:
            self.F += d_gfu*alpha/h*u_bnd_gfu*test*ds(definedon = self.input_params['Dir_bnd'], skeleton = True, deformation = deform)\
                - d_gfu*InnerProduct(n, grad(test))*u_bnd_gfu*ds(definedon = self.input_params['Dir_bnd'], skeleton=True, deformation = deform)
        if self.input_params['Neu_bnd']:
            self.F += d_gfu*gradu_gfu*n*test*ds(definedon = self.input_params['Neu_bnd'], deformation = deform)

        self.F += -IfPos(b_gfu*n, CF(0), b_gfu*n*u_bnd_gfu)*test*ds(deformation = deform)

        self.alpha1 = Parameter(1.0)
        self.F += self.alpha1/self._solver.time.dt*self.gfu_old*test*dx(deformation = deform_old)
        self.alpha2 = Parameter(0.0)
        self.F += self.alpha2/self._solver.time.dt*self.gfu_oldold*test*dx(deformation = deform_oldold)

    def PreProcess(self):
        
        self.gfu_oldold.vec.data = self.gfu_old.vec.data
        self.gfu_old.vec.data = self.gfu.vec.data

        if self._solver.time.iter==0 and self.input_params["mass_preserving"]:
            gfu0_vec = self.gfu.vec.Copy().FV().NumPy()
            self.mass0 = np.sum(self.weights*gfu0_vec)

        if self._solver.time.iter == 1:
            self.alpha0.Set(1.5)
            self.alpha1.Set(2)
            self.alpha2.Set(-0.5)

    def Solve(self):

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
        
        pass

    @property
    def sol(self):
        return self.gfu
    
    @sol.setter
    def sol(self, cf):
        self.gfu.Set(cf, definedon = self.domain)

        