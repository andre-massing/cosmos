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
        self.params["conservative"] = False
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
        if self.model.dim == 2:
            self.gfu_vtk = [GridFunction(H1(self.model.parentmesh, order = self.params["fes_order"])) for i in range(self.sys_dim)]
        else:
            self.gfu_vtk = self.sol

        self.sol_old = [self.gfu_old.components[2*i] for i in range(self.sys_dim)]
        for i in range(self.sys_dim):
            self.output_fields["sol_" + str(i+1)] = OutputField(self.gfu.components[2*i], "sol_" + str(i+1), BND)

        self.vectorspace= VectorH1(model.parentmesh, order = 1, definedon = compartment.domain)
        self.Amap_h = GridFunction(self.vectorspace)
        self.discscalarspace = SurfaceL2(model.parentmesh, order = 0, definedon =compartment.domain)
        self.J_h_old = GridFunction(self.discscalarspace)
        if model.dim == 2:
            self.identity = CF((x,y))
        else:
            self.identity = CF((x,y,z))

        self.mass0 = np.zeros(self.sys_dim)

    def Initialize(self):

        self.dX_new = GridFunction(self.model.dX.space)
        self.dX_old = GridFunction(self.model.dX.space)

        ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])

        for i in range(self.sys_dim):
            self.output_fields["sol_" + str(i+1)]._coef.Set(self.params["u0_" + str(i+1)], definedon = self.compartment.domain, dual = True)

            if self.params["mass_preserving_" + str(i+1)] and self.params["fes_order"]>1:
                raise Exception('Mass preservation not yet implemented for fes_order>1')
            if self.params["bounds_" + str(i+1)] and self.params["fes_order"]>1:
                raise Exception('Bounds preservation not yet implemented for fes_order>1')
            
            if self.params["mass_preserving_" + str(i+1)]:
                ds_lumped = ds(intrules = {  SEGM : ir_segm, TRIG : ir_trig }, deformation = self.dX_new)
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
            if self.model.dim == 2:
                gfu_one = GridFunction(H1(self.model.parentmesh, order = 1))
                gfu_one.Set(1, definedon = self.compartment.domain)
            else:
                gfu_one = GridFunction(H1(self.model.parentmesh, order = 1, definedon = self.compartment.domain))
                gfu_one.Set(1, definedon = self.compartment.domain)
            self.vtk = VTKOutput(self.model.parentmesh,
                                coefs=[self.gfu_vtk[i] for i in range(self.sys_dim)] + [gfu_one],
                                names =['concentration_' + str(i+1) for i in range(self.sys_dim)] + ['indicator'],
                                filename= output_vtk_name, 
                                subdivision = self.params['subdivision'])

    def PreProcess(self):
        
        self.gfu_old.vec.data = self.gfu.vec.data

        self.ns = specialcf.normal(self.model.dim)
        # self.Amap_h.Set(self.identity - self.model.dt*self.model.ale.ale_velocity, dual = True, definedon =self.compartment.domain)
        # self.J_h_old.Set(Det(Grad(self.Amap_h).Trace().trans*Grad(self.Amap_h).Trace() + OuterProduct(self.ns, self.ns)), definedon=self.compartment.domain)

        self.dX_new.vec.data = 2*self.model.dX.vec.data - self.model.ale.prev_dX[-2].data
        self.dX_old.vec.data = self.model.dX.vec.data

    def Solve(self):

        trial, test = self.fes.TnT()
        self.A = BilinearForm(self.fes)
        self.F = LinearForm(self.fes)
        Ps = Id(self.model.dim) - OuterProduct(self.ns, self.ns)
        h = self.cfg.h
        alpha = 5 * self.params["fes_order"] * (self.params["fes_order"]+1)
        tE = specialcf.tangential(self.model.dim)
        if self.model.dim == 2:
            facet_space = self.V
            nE = specialcf.tangential(self.model.dim)
        else:
            facet_space = FacetSurface(self.model.parentmesh, order = 0)
            nE = Cross(self.ns, tE)

        ############### Do I actually need this? Getting rid of it for now
        ############### might be a problem for manufactured solutions
        # self.model.time.helper.t.Set(self.model.time.t.Get() + self.model.time.dt.Get())

        for i in range(self.sys_dim):

            c = self.params["c_" + str(i+1)]().Compile()
            d = self.params["d_" + str(i+1)]().Compile()
            b = self.params["b_" + str(i+1)]().Compile()
            rhs = self.params["rhs_" + str(i+1)]().Compile()
            u_bnd = self.params["u_bnd_" + str(i+1)]().Compile()
            gradu_bnd = self.params["gradu_bnd_" + str(i+1)]().Compile()

            self.A += c*trial[2*i]*test[2*i]*ds(deformation = self.dX_new)
            self.A += d*grad(trial[2*i]).Trace()*grad(test[2*i]).Trace()*ds(deformation = self.dX_new)
                    
            if self.params['Dir_bnd']:
                dir_bnd_gfu = GridFunction(facet_space)
                dir_bnd_gfu.Set(1, definedon = self.model.parentmesh.BBoundaries(self.params['Dir_bnd']))
                self.A += - dir_bnd_gfu*d*InnerProduct(nE, grad(trial[2*i]).Trace())*test[2*i]*ds(element_boundary=True, deformation = self.dX_new) \
                    - dir_bnd_gfu*d*InnerProduct(nE, grad(test[2*i]).Trace())*trial[2*i]*ds(element_boundary=True, deformation = self.dX_new)\
                    + dir_bnd_gfu*d*alpha/h*trial[2*i]*test[2*i]*ds(element_boundary=True, deformation = self.dX_new)

            self.A += -b*grad(test[2*i]).Trace() * trial[2*i]*ds(deformation = self.dX_new)
            bnd_gfu = GridFunction(facet_space)
            bnd_gfu.Set(1, definedon = self.model.parentmesh.BBoundaries(self.params['Dir_bnd']+'|'+self.params['Neu_bnd']))
            self.A += bnd_gfu*IfPos(b*nE, b*nE*trial[2*i], CF(0))*test[2*i]\
                *ds(element_boundary=True, deformation = self.dX_new)
            
            if self.model.dim == 2:
                tEc = CF((-self.ns[1], self.ns[0]))
                jump_dudn = (trial[2*i].Trace().Deriv() - trial[2*i +1]*tEc)*nE
                jump_dvdn = (test[2*i].Trace().Deriv() - test[2*i +1]*tEc)*nE
            elif self.model.dim == 3:
                jump_dudn = (trial[2*i].Trace().Deriv() - trial[2*i +1].Trace())*nE
                jump_dvdn = (test[2*i].Trace().Deriv() - test[2*i +1].Trace())*nE
            stab = Norm(b)*h**2
            self.A +=  IfPos(stab, stab*InnerProduct(jump_dudn,jump_dvdn), InnerProduct(trial[2*i +1].Trace(),test[2*i +1].Trace()) )\
                    *ds(element_boundary=True, deformation = self.dX_new)
            self.A +=  -1*bnd_gfu*IfPos(stab, stab*InnerProduct(jump_dudn,jump_dvdn), InnerProduct(trial[2*i +1].Trace(),test[2*i +1].Trace()) )\
                    *ds(element_boundary=True, deformation = self.dX_new)
            self.A +=  bnd_gfu*InnerProduct(trial[2*i +1].Trace(),test[2*i +1].Trace())\
                    *ds(element_boundary=True, deformation = self.dX_new)

            self.F += rhs*test[2*i]*ds(deformation = self.dX_new)
            
            if self.params['Dir_bnd']:
                self.F += dir_bnd_gfu*d*alpha/h*u_bnd*test[2*i]*ds(element_boundary=True, deformation = self.dX_new)\
                    - dir_bnd_gfu*d*InnerProduct(nE, grad(test[2*i]).Trace())*u_bnd*ds(element_boundary=True, deformation = self.dX_new)
            if self.params['Neu_bnd']:
                neu_bnd_gfu = GridFunction(facet_space)
                neu_bnd_gfu.Set(1, definedon = self.model.parentmesh.BBoundaries(self.params['Neu_bnd']))
                self.F += neu_bnd_gfu*d*gradu_bnd*nE*test[2*i]*ds(element_boundary=True, deformation = self.dX_new)

            self.F += -bnd_gfu*IfPos(b*nE, CF(0), b*nE*u_bnd)*test[2*i]*ds(element_boundary=True, deformation = self.dX_new)
            
            self.A += 1/self.model.dt*trial[2*i]*test[2*i]*ds(deformation = self.dX_new)
            self.F += 1/self.model.dt*self.gfu_old.components[2*i]*test[2*i]*ds(deformation = self.dX_old)

            ########### Time derivative!!!!
            # if not self.params['conservative']:
            #     self.A += 1/self.model.dt*trial[2*i]*test[2*i]*ds(deformation = self.model.dX)
            #     self.F += self.J_h_old/self.model.dt*self.gfu_old.components[2*i]*test[2*i]*ds(deformation = self.model.dX)
            # elif self.params['conservative']:
            #     self.A += 1/self.model.dt*trial[2*i]*test[2*i]*ds(deformation = self.model.dX)
            #     self.F += 1/self.model.dt*self.gfu_old.components[2*i]*test[2*i]*ds(deformation = self.model.dX)

            #     self.dX_ab.vec.data = 1.5*self.model.dX.vec.data - 0.5*self.model.ale.prev_dX[-2].data
            #     self.dX_b.vec.data = 2*self.model.dX.vec.data - self.model.ale.prev_dX[-2].data
            #     self.A += 1/6*trial[2*i]*test[2*i]*Trace(Grad(self.model.ale.ale_velocity).Trace())*ds(deformation = self.model.dX)
            #     self.A += 2/3*trial[2*i]*test[2*i]*Trace(Grad(self.model.ale.ale_velocity).Trace())*ds(deformation = self.dX_ab)
            #     self.A += 1/6*trial[2*i]*test[2*i]*Trace(Grad(self.model.ale.ale_velocity).Trace())*ds(deformation = self.dX_b)

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = self.fes.FreeDofs())
        self.F.Assemble()

        ############### Do I actually need this? Getting rid of it for now
        ############### might be a problem for manufactured solutions
        # self.model.time.helper.t.Set(self.model.time.t.Get())

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

        if self.model.dim == 2:
            for i, gfu in enumerate(self.gfu_vtk):
                gfu.Set(self.sol[i], definedon = self.compartment.domain)

        del self.A
        del self.invA
        del self.F