import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.field import InputField, OutputField, Field
from cosmos.core.compartment import CosmosCompartment
from cosmos.core.model import CosmosModel
from cosmos.core.utils import MandBP
from ngsolve.webgui import Draw
import numbers
import time

class ADRVolumeSystemBDF1Model(BasePDEModel):

    def __init__(self, name:str = 'ADRVolumeSystemBDF1Model', model:CosmosModel = None, compartment:CosmosCompartment = None, **kwargs):
        
        super().__init__(name=name, model = model, compartment=compartment)
        
        self.is_bnd = False
        self.is_vol = True
        self.name = name
        self.model = model
        self.compartment = compartment

        if 'dim' in kwargs.keys():
            if isinstance(kwargs['dim'], numbers.Number):
                self.sys_dim = kwargs['dim']
            else:
                raise Exception(f'Dimension of systems {self.name} must be a number')
        else:
            raise Exception('Parameter dim is needed for initialization of ADRBoundarySystemBDF1Model')
        

        self.params["Neu_bnd"] = ''
        self.params["Dir_bnd"] = ''
        self.params["fes_order"] = 1
        self.params["subdivision"] = 0

        V = H1(model.parentmesh, order = self.params["fes_order"], 
                                definedon = compartment.domain, dgjumps = True)
        self.fes = V
        for i in range(self.sys_dim-1):
            self.fes = self.fes*V
        
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)
        self.sol = self.gfu.components
        for i in range(self.sys_dim):
            self.output_fields["sol_" + str(i+1)] = OutputField(self.gfu.components[i], "sol_" + str(i+1), VOL)

        self.deform = GridFunction(self.model.dX.space)
        self.deform.vec.data = self.model.dX.vec.data

        self.params["ale_velocity"] = Field(CF((0,)*compartment.dim_emd))
        for i in range(self.sys_dim):
            self.params["mass_preserving_" + str(i+1)] = False
            self.params["bounds_" + str(i+1)] = None
            self.params["u0_" + str(i+1)] = CF(0)
            self.params["b_" + str(i+1)] = Field(CF((0,)*compartment.dim_emd))
            self.params["d_" + str(i+1)] = Field(CF(0))
            self.params["c_" + str(i+1)] = Field(CF(0))
            self.params["rhs_" + str(i+1)] = Field(CF(0))
            self.params["gradu_bnd_" + str(i+1)] = Field(CF((0,)*compartment.dim_emd))
            self.params["u_bnd_" + str(i+1)] = Field(CF(0))

        self.mass0 = []

    def Initialize(self):

        ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
        ir_tet = IntegrationRule(points  = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], \
                                        weights = [1/24, 1/24, 1/24, 1/24])

        for i in range(self.sys_dim):
            self.output_fields["sol_" + str(i+1)]._coef.Set(self.params["u0_" + str(i+1)], definedon = self.compartment.domain, dual = True)

            if self.params["mass_preserving_" + str(i+1)] and self.params["fes_order_" + str(i+1)]>1:
                raise Exception('Mass preservation not yet implemented for fes_order>1')
            if self.params["bounds_" + str(i+1)] and self.params["fes_order_" + str(i+1)]>1:
                raise Exception('Bounds preservation not yet implemented for fes_order>1')
            
            if self.params["mass_preserving_" + str(i+1)]:
                dx_lumped = dx(intrules = {  TRIG : ir_trig , TET : ir_tet }, deformation = self.deform)
                self.Amp = BilinearForm(self.output_fields["sol_" + str(i+1)]._coef, symmetric = True)
                u, v = self.output_fields["sol_" + str(i+1)]._coef.space.TnT()
                self.Amp += u*v*dx_lumped
                self.Amp.Assemble()
                rows,cols,vals = self.Amp.mat.COO()
                weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
                gfu0_vec = self.gfu.vec.Copy().FV().NumPy()
                self.mass0[i] = np.sum(weights*gfu0_vec)
            
            if self.params["bounds_" + str(i+1)]:
                self.output_fields["sol_" + str(i+1)]._coef.vec.data[:] = np.clip(self.output_fields["sol_" + str(i+1)]._coef.vec.FV().NumPy(),
                                                                                  self.params["bounds_" + str(i+1)][0], self.params["bounds_" + str(i+1)][1])

        if self.model.io.root != None:
            output_vtk_folder = os.path.join(self.model.io.root, self.name)
            output_vtk_name = os.path.join(output_vtk_folder, self.name)
            os.makedirs(output_vtk_folder, exist_ok=True)
            if self.compartment.dim == 2:
                self.vtk = VTKOutput(self.model.parentmesh,
                                    coefs=[self.output_fields["sol_" + str(i+1)]._coef for i in range(self.sys_dim)],
                                    names =['concentration_' + str(i+1) for i in range(self.sys_dim)],
                                    filename= output_vtk_name, 
                                    subdivision = self.params['subdivision'])
            else:
                raise Exception('Not yet implemented!')

    def PreProcess(self):
        
        self.gfu_old.vec.data = self.gfu.vec.data

    def Solve(self):

        trial, test = self.fes.TnT()
        self.A = BilinearForm(self.fes)
        self.F = LinearForm(self.fes)

        n = specialcf.normal(self.model.dim)
        h = self.cfg.h
        alpha = 5 * self.params["fes_order"] * (self.params["fes_order"]+1)

        ##############################################
        # AAA this has to be updated carefully in the new version
        self.deform.Set(self.model.dX + self.model.dt*self.params['ale_velocity']().Compile(), dual = True, definedon = self.compartment.domain)
        deform_old = self.model.dX 
        ##############################################

        self.model.time.helper.t.Set(self.model.time.t.Get() + self.model.time.dt.Get())

        for i in range(self.sys_dim):

            c = self.params["c_" + str(i+1)]().Compile()
            d = self.params["d_" + str(i+1)]().Compile()
            b = self.params["b_" + str(i+1)]().Compile()
            rhs = self.params["rhs_" + str(i+1)]().Compile()
            u_bnd = self.params["u_bnd_" + str(i+1)]().Compile()
            gradu_bnd = self.params["gradu_bnd_" + str(i+1)]().Compile()

            self.A += c*trial[i]*test[i]*dx(deformation =self.deform)
            self.A += d*grad(trial[i])*grad(test[i])*dx(deformation =self.deform)
                    
            if self.params['Dir_bnd']:
                self.A += - d*InnerProduct(n, grad(trial[i]))*test[i]*ds(definedon = self.params['Dir_bnd'], skeleton=True, deformation =self.deform) \
                    - d*InnerProduct(n, grad(test[i]))*trial[i]*ds(definedon = self.params['Dir_bnd'], skeleton=True, deformation =self.deform)\
                    + d*alpha/h*trial[i]*test[i]*ds(definedon = self.params['Dir_bnd'], skeleton = True, deformation =self.deform)\

            self.A += -b*grad(test[i]) * trial[i]*dx(deformation =self.deform)
            stab = (Norm(b))*h**2
            jump_u = grad(trial[i])-grad(trial[i]).Other()
            jump_v = grad(test[i])-grad(test[i]).Other()
            self.A += stab*jump_u*jump_v*dx(deformation = self.deform, skeleton = True)
            self.A += IfPos(b*n, b*n*trial[i], CF(0))*test[i]\
                *ds(deformation =self.deform)
            
            self.A += 1/self.model.dt*trial[i]*test[i]*dx(deformation =self.deform)

            self.F += rhs*test[i]*dx(deformation =self.deform)
            
            if self.params['Dir_bnd']:
                self.F += d*alpha/h*u_bnd*test[i]*ds(definedon = self.params['Dir_bnd'], skeleton = True, deformation =self.deform)\
                    - d*InnerProduct(n, grad(test[i]))*u_bnd*ds(definedon = self.params['Dir_bnd'], skeleton=True, deformation =self.deform)
            if self.params['Neu_bnd']:
                self.F += d*gradu_bnd*n*test[i]*ds(definedon = self.params['Neu_bnd'], deformation =self.deform)

            self.F += -IfPos(b*n, CF(0), b*n*u_bnd)*test[i]*ds(deformation =self.deform)
            self.F += 1/self.model.dt*self.gfu_old.components[i]*test[i]*dx(deformation = deform_old)
        
        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = self.fes.FreeDofs())
        self.F.Assemble()

        self.model.time.helper.t.Set(self.model.time.t.Get())

        self.gfu.vec.data = self.invA*self.F.vec

        for i in range(self.sys_dim):

            if self.params["bounds_" + str(i+1)] and not self.params["mass_preserving_" + str(i+1)]:

                gfu_vec = self.output_fields["sol_" + str(i+1)]._coef.vec.Copy().FV().NumPy()
                gfu_new = MandBP(gfu_vec, BP = self.params["bounds"])
                self.output_fields["sol_" + str(i+1)]._coef.vec.data = gfu_new

            elif self.params["mass_preserving_" + str(i+1)]:

                if hasattr(self.model.time, 'dt'):
                    dt = self.model.dt.Get()
                else:
                    logger.error('A time-dependent simulation is needed to impose conservative mass')

                self.Amp.Assemble()
                rows,cols,vals = self.Amp.mat.COO()
                weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
                gfu_vec = self.output_fields["sol_" + str(i+1)]._coef.vec.Copy().FV().NumPy()

                if self.params["bounds_" + str(i+1)]:
                    BP = self.params["bounds_" + str(i+1)]
                else:
                    BP = [-np.inf, np.inf]

                gfu_new = MandBP(gfu_vec, weights=weights, BP=BP,
                                    MP=self.params["mass_preserving_" + str(i+1)], mass0=self.mass0[i], dt = dt)

                self.output_fields["sol_" + str(i+1)]._coef.vec.data = gfu_new

    def PostProcess(self):

        pass