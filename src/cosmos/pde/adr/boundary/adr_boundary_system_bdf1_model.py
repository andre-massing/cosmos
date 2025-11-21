import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.field import OutputField, Field
from cosmos.core.compartment import CosmosCompartment
from cosmos.core.model import CosmosModel
from cosmos.core.utils import MandBP
from ngsolve.webgui import Draw
import numbers
import time

class ADRBoundarySystemBDF1Model(BasePDEModel):

    def __init__(self, name:str = 'ADRBoundarySystemBDF1Model', model:CosmosModel = None, compartment:CosmosCompartment = None, **kwargs):
        
        super().__init__(name=name, model = model, compartment=compartment)
        
        self.is_bnd = True
        self.is_vol = False
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
        self.params["subdivision"] = 0
        self.params["fes_order"] = 1
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

        self.V = H1(model.parentmesh, order = self.params["fes_order"], 
                                definedon = compartment.domain)
        if self.model.dim == 2:
            dV = H1(self.model.parentmesh, order = self.params["fes_order"],
                definedon=compartment.domain)
        else:
            dV = NormalFacetSurface(self.model.parentmesh, order = self.params["fes_order"]-1,
                definedon=compartment.domain)
            
        self.fes = self.V*dV
        for i in range(self.sys_dim-1):
            self.fes = self.fes*self.V*dV
        
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)

        self.sol = [self.gfu.components[2*i] for i in range(self.sys_dim)]
        for i in range(self.sys_dim):
            self.output_fields["sol_" + str(i+1)] = OutputField(self.gfu.components[2*i], "sol_" + str(i+1), BND)

        self.deform = GridFunction(self.model.dX.space)
        self.deform.vec.data = self.model.dX.vec.data

        self.mass0 = np.zeros(self.sys_dim)
        

    def Initialize(self):

        ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])

        for i in range(self.sys_dim):
            self.output_fields["sol_" + str(i+1)]._coef.Set(self.params["u0_" + str(i+1)], definedon = self.compartment.domain, dual = True)

            if self.params["mass_preserving_" + str(i+1)] and self.params["fes_order"]>1:
                raise Exception('Mass preservation not yet implemented for fes_order>1')
            if self.params["bounds_" + str(i+1)] and self.params["fes_order"]>1:
                raise Exception('Bounds preservation not yet implemented for fes_order>1')
            
            if self.params["mass_preserving_" + str(i+1)]:
                ds_lumped = ds(intrules = {  SEGM : ir_segm, TRIG : ir_trig }, deformation = self.deform)
                self.Amp = BilinearForm(self.output_fields["sol_" + str(i+1)]._coef.space, symmetric = True)
                u, v = self.output_fields["sol_" + str(i+1)]._coef.space.TnT()
                self.Amp += u*v*ds_lumped
                self.Amp.Assemble()
                rows,cols,vals = self.Amp.mat.COO()
                weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
                gfu0_vec = self.output_fields["sol_" + str(i+1)]._coef.vec.Copy().FV().NumPy()
                self.mass0[i] = np.sum(weights*gfu0_vec)
            
            if self.params["bounds_" + str(i+1)]:
                self.output_fields["sol_" + str(i+1)]._coef.vec.data[:] = np.clip(self.output_fields["sol_" + str(i+1)]._coef.vec.FV().NumPy(), 
                                                                                  self.params["bounds_" + str(i+1)][0], self.params["bounds_" + str(i+1)][1])

        if self.model.io.root != None:
            output_vtk_folder = os.path.join(self.model.io.root, self.name)
            output_vtk_name = os.path.join(output_vtk_folder, self.name)
            os.makedirs(output_vtk_folder, exist_ok=True)
            self.vtk = VTKOutput(self.model.parentmesh,
                                coefs=[self.output_fields["sol_" + str(i+1)]._coef for i in range(self.sys_dim)],
                                names =['concentration_' + str(i+1) for i in range(self.sys_dim)],
                                filename= output_vtk_name, 
                                subdivision = self.params['subdivision'])

    def PreProcess(self):
        
        self.gfu_old.vec.data = self.gfu.vec.data

        trial, test = self.fes.TnT()
        self.A = BilinearForm(self.fes)
        self.F = LinearForm(self.fes)

        n = specialcf.normal(self.model.dim)
        Ps = Id(self.model.dim) - OuterProduct(n, n)
        h = self.cfg.h
        alpha = 5 * self.params["fes_order"] * (self.params["fes_order"]+1)
        tE = specialcf.tangential(self.model.dim)
        if self.model.dim == 2:
            facet_space = self.V
            nE = specialcf.tangential(self.model.dim)
        else:
            facet_space = FacetSurface(self.model.parentmesh, order = 0)
            nE = Cross(n, tE)

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

            self.A += c*trial[2*i]*test[2*i]*ds(deformation = self.deform)
            self.A += d*grad(trial[2*i]).Trace()*grad(test[2*i]).Trace()*ds(deformation = self.deform)
                    
            if self.params['Dir_bnd']:
                dir_bnd_gfu = GridFunction(facet_space)
                dir_bnd_gfu.Set(1, definedon = self.model.parentmesh.BBoundaries(self.params['Dir_bnd']))
                self.A += - dir_bnd_gfu*d*InnerProduct(nE, grad(trial[2*i]).Trace())*test[2*i]*ds(element_boundary=True, deformation = self.deform) \
                    - dir_bnd_gfu*d*InnerProduct(nE, grad(test[2*i]).Trace())*trial[2*i]*ds(element_boundary=True, deformation = self.deform)\
                    + dir_bnd_gfu*d*alpha/h*trial[2*i]*test[2*i]*ds(element_boundary=True, deformation = self.deform)

            self.A += -b*grad(test[2*i]).Trace() * trial[2*i]*ds(deformation = self.deform)
            bnd_gfu = GridFunction(facet_space)
            bnd_gfu.Set(1, definedon = self.model.parentmesh.BBoundaries(self.params['Dir_bnd']+'|'+self.params['Neu_bnd']))
            self.A += bnd_gfu*IfPos(b*nE, b*nE*trial[2*i], CF(0))*test[2*i]\
                *ds(element_boundary=True, deformation = self.deform)
            
            if self.model.dim == 2:
                tEc = CF((-n[1], n[0]))
                jump_dudn = (trial[2*i].Trace().Deriv() - trial[2*i +1]*tEc)*nE
                jump_dvdn = (test[2*i].Trace().Deriv() - test[2*i +1]*tEc)*nE
            elif self.model.dim == 3:
                jump_dudn = (trial[2*i].Trace().Deriv() - trial[2*i +1].Trace())*nE
                jump_dvdn = (test[2*i].Trace().Deriv() - test[2*i +1].Trace())*nE
            stab = Norm(b)*h**2
            self.A +=  IfPos(stab, stab*InnerProduct(jump_dudn,jump_dvdn), InnerProduct(trial[2*i +1].Trace(),test[2*i +1].Trace()) )\
                    *ds(element_boundary=True, deformation = self.deform)
            # self.A +=  stab*InnerProduct(jump_dudn,jump_dvdn)\
            #         *ds(element_boundary=True, deformation = self.deform)
            self.A +=  -1*stab*bnd_gfu*InnerProduct(jump_dudn,jump_dvdn)\
                    *ds(element_boundary=True, deformation = self.deform)
            self.A +=  bnd_gfu*InnerProduct(trial[2*i +1].Trace(),test[2*i +1].Trace())\
                    *ds(element_boundary=True, deformation = self.deform)
            
            self.A += 1/self.model.dt*trial[2*i]*test[2*i]*ds(deformation = self.deform)

            self.F += rhs*test[2*i]*ds(deformation = self.deform)
            
            if self.params['Dir_bnd']:
                self.F += dir_bnd_gfu*d*alpha/h*u_bnd*test[2*i]*ds(element_boundary=True, deformation = self.deform)\
                    - dir_bnd_gfu*d*InnerProduct(nE, grad(test[2*i]).Trace())*u_bnd*ds(element_boundary=True, deformation = self.deform)
            if self.params['Neu_bnd']:
                neu_bnd_gfu = GridFunction(facet_space)
                neu_bnd_gfu.Set(1, definedon = self.model.parentmesh.BBoundaries(self.params['Neu_bnd']))
                self.F += neu_bnd_gfu*d*gradu_bnd*nE*test[2*i]*ds(element_boundary=True, deformation = self.deform)

            self.F += -bnd_gfu*IfPos(b*nE, CF(0), b*nE*u_bnd)*test[2*i]*ds(element_boundary=True, deformation = self.deform)

            self.F += 1/self.model.dt*self.gfu_old.components[2*i]*test[2*i]*ds(deformation = deform_old)

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = self.fes.FreeDofs())
        self.F.Assemble()

        self.model.time.helper.t.Set(self.model.time.t.Get())

    def Solve(self):

        self.gfu.vec.data = self.invA*self.F.vec

        for i in range(self.sys_dim):

            if self.params["bounds_" + str(i+1)] and not self.params["mass_preserving_" + str(i+1)]:

                gfu_vec = self.output_fields["sol_" + str(i+1)]._coef.vec.Copy().FV().NumPy()
                gfu_new = MandBP(gfu_vec, BP = self.params["bounds_" + str(i+1)])
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