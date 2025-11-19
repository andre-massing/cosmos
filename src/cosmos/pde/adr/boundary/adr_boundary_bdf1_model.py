import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.field import InputField, OutputField
from cosmos.core.compartment import CosmosCompartment
from cosmos.core.model import CosmosModel
from cosmos.core.utils import MandBP
from ngsolve.webgui import Draw
import time

class ADRBoundaryBDF1Model(BasePDEModel):

    def __init__(self, name:str = 'ADRBoundaryBDF1Model', model:CosmosModel = None, compartment:CosmosCompartment = None):
        
        super().__init__(name=name, model = model, compartment=compartment)
        
        self.is_bnd = True
        self.is_vol = False
        self.name = name
        self.model = model
        self.compartment = compartment

        self.params["Neu_bnd"] = ''
        self.params["Dir_bnd"] = ''
        self.params["mass_preserving"] = False
        self.params["bounds"] = None
        self.params["fes_order"] = 1
        self.params["u0"] = CF(0)
        self.params["subdivision"] = 0

        self.fes = H1(model.parentmesh, order = self.params["fes_order"], 
                                definedon = compartment.domain)
        fes_vector = VectorH1(model.parentmesh, order = self.params["fes_order"], 
                                definedon = compartment.domain)
        
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)
        self.output_fields["sol"] = OutputField(self.gfu, "sol", BND)

        self.deform = GridFunction(self.model.dX.space)
        self.deform.vec.data = self.model.dX.vec.data

        # Creating GridFunctions for the Fields
        self.b_gfu = GridFunction(fes_vector)
        self.d_gfu = GridFunction(self.fes)
        self.c_gfu = GridFunction(self.fes)
        self.rhs_gfu = GridFunction(self.fes)
        self.gradu_gfu = GridFunction(fes_vector)
        self.u_bnd_gfu = GridFunction(self.fes)
        self.ale_velocity_gfu = GridFunction(fes_vector)
        self.params["b"] = InputField(self.b_gfu, CF((0,)*compartment.dim_emd), "b", self.compartment.domain)
        self.params["d"] = InputField(self.d_gfu, CF(0), "d", self.compartment.domain)
        self.params["c"] = InputField(self.c_gfu, CF(0), "c", self.compartment.domain)
        self.params["rhs"] = InputField(self.rhs_gfu, CF(0), "rhs", self.compartment.domain)
        self.params["gradu_bnd"] = InputField(self.gradu_gfu, CF((0,)*compartment.dim_emd), "gradu_bnd", self.compartment.boundary)
        self.params["u_bnd"] = InputField(self.u_bnd_gfu, CF(0), "u_bnd", self.compartment.boundary)
        self.params["ale_velocity"] = InputField(self.ale_velocity_gfu, CF((0,)*compartment.dim_emd), "ale_velocity", self.compartment.domain)

    def Initialize(self):

        self.gfu.Set(self.params['u0'], definedon = self.compartment.domain, dual = True)

        if self.params["mass_preserving"] and self.params["fes_order"]>1:
            raise Exception('Mass preservation not yet implemented for fes_order>1')
        if self.params["bounds"] and self.params["fes_order"]>1:
            raise Exception('Bounds preservation not yet implemented for fes_order>1')
        
        if self.params["mass_preserving"]:
            ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = {  SEGM : ir_segm, TRIG : ir_trig }, deformation = self.deform)
            self.Amp = BilinearForm(self.gfu.space, symmetric = True)
            u, v = self.gfu.space.TnT()
            self.Amp += u*v*ds_lumped
            self.Amp.Assemble()
            rows,cols,vals = self.Amp.mat.COO()
            weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
            gfu0_vec = self.gfu.vec.Copy().FV().NumPy()
            self.mass0 = np.sum(weights*gfu0_vec)
        
        if self.params["bounds"]:
            self.gfu.vec.data[:] = np.clip(self.gfu.vec.FV().NumPy(), self.params["bounds"][0], self.params["bounds"][1])

        if self.model.io.root != None:
            output_vtk_folder = os.path.join(self.model.io.root, self.name)
            output_vtk_name = os.path.join(output_vtk_folder, self.name)
            os.makedirs(output_vtk_folder, exist_ok=True)
            if self.compartment.dim == 2:
                self.vtk = VTKOutput(self.model.parentmesh,
                                    coefs=[self.gfu],
                                    names =['concentration'],
                                    filename= output_vtk_name, 
                                    subdivision = self.params['subdivision'])
            else:
                raise Exception('Not yet implemented!')

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
            facet_space = self.fes
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

        c = self.params["c"]().Compile()
        d = self.params["d"]().Compile()
        b = self.params["b"]().Compile()
        rhs = self.params["rhs"]().Compile()
        u_bnd = self.params["u_bnd"]().Compile()
        gradu_bnd = self.params["gradu_bnd"]().Compile()

        self.A += c*trial*test*ds(deformation = self.deform)
        self.A += d*grad(trial).Trace()*grad(test).Trace()*ds(deformation = self.deform)
                
        if self.params['Dir_bnd']:
            dir_bnd_gfu = GridFunction(facet_space)
            dir_bnd_gfu.Set(1, definedon = self.model.parentmesh.BBoundaries(self.params['Dir_bnd']))
            self.A += - dir_bnd_gfu*d*InnerProduct(nE, grad(trial).Trace())*test*ds(element_boundary=True, deformation = self.deform) \
                - dir_bnd_gfu*d*InnerProduct(nE, grad(test).Trace())*trial*ds(element_boundary=True, deformation = self.deform)\
                + dir_bnd_gfu*d*alpha/h*trial*test*ds(element_boundary=True, deformation = self.deform)

        self.A += -b*grad(test).Trace() * trial*ds(deformation = self.deform)
        bnd_gfu = GridFunction(facet_space)
        bnd_gfu.Set(1, definedon = self.model.parentmesh.BBoundaries(self.params['Dir_bnd']+'|'+self.params['Neu_bnd']))
        self.A += bnd_gfu*IfPos(b*nE, b*nE*trial, CF(0))*test\
            *ds(element_boundary=True, deformation = self.deform)
        
        self.A += 1/self.model.dt*trial*test*ds(deformation = self.deform)

        self.F += rhs*test*ds(deformation = self.deform)
        
        if self.params['Dir_bnd']:
            self.F += dir_bnd_gfu*d*alpha/h*u_bnd*test*ds(element_boundary=True, deformation = self.deform)\
                - dir_bnd_gfu*d*InnerProduct(nE, grad(test).Trace())*u_bnd*ds(element_boundary=True, deformation = self.deform)
        if self.params['Neu_bnd']:
            neu_bnd_gfu = GridFunction(facet_space)
            neu_bnd_gfu.Set(1, definedon = self.model.parentmesh.BBoundaries(self.params['Neu_bnd']))
            self.F += neu_bnd_gfu*d*gradu_bnd*nE*test*ds(element_boundary=True, deformation = self.deform)

        self.F += -bnd_gfu*IfPos(b*nE, CF(0), b*nE*u_bnd)*test*ds(element_boundary=True, deformation = self.deform)

        self.F += 1/self.model.dt*self.gfu_old*test*ds(deformation = deform_old)

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = self.fes.FreeDofs())
        self.F.Assemble()

        self.model.time.helper.t.Set(self.model.time.t.Get())

    def Solve(self):

        self.gfu.vec.data = self.invA*self.F.vec

        if self.params["bounds"] and not self.params["mass_preserving"]:

            gfu_vec = self.gfu.vec.Copy().FV().NumPy()
            gfu_new = MandBP(gfu_vec, BP = self.params["bounds"])
            self.gfu.vec.data = gfu_new

        elif self.params["mass_preserving"]:

            if hasattr(self.model.time, 'dt'):
                dt = self.model.dt.Get()
            else:
                logger.error('A time-dependent simulation is needed to impose conservative mass')

            self.Amp.Assemble()
            rows,cols,vals = self.Amp.mat.COO()
            weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
            gfu_vec = self.gfu.vec.Copy().FV().NumPy()

            if self.params["bounds"]:
                BP = self.params["bounds"]
            else:
                BP = [-np.inf, np.inf]

            gfu_new = MandBP(gfu_vec, weights=weights, BP=BP,
                                MP=self.params["mass_preserving"], mass0=self.mass0, dt = dt)

            self.gfu.vec.data = gfu_new

    def PostProcess(self):

        pass