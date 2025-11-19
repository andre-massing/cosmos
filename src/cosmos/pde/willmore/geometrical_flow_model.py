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
import time

class GeometricalFlowModel(BasePDEModel):

    def __init__(self, name:str = 'GeometricalFlow', model:CosmosModel = None, compartment:CosmosCompartment = None):
        
        super().__init__(name=name, model = model, compartment=compartment)
        
        self.is_bnd = True
        self.is_vol = False
        self.name = name
        self.model = model
        self.compartment = compartment

        self.cn_bboundary = compartment.clamped_bbnd + '|' + compartment.navier_bbnd
        self.navier_bbnd = compartment.navier_bbnd
        self.params["subdivision"] = 0
        self.params["sp_curv"] = CF(0)
        self.params["rhs"] = Field(CF(0))
        self.params["alpha"] = CF(1)
        self.params["beta"] = CF(0)
        self.params["gamma"] = CF(0)

        self.vectorspace_bbnd = VectorH1(model.parentmesh, order = 1, definedon = compartment.domain, dirichlet_bbnd = self.cn_bboundary)
        self.vectorspace = VectorH1(model.parentmesh, order = 1, definedon =compartment.domain)
        self.scalarspace_bbnd = H1(model.parentmesh, order = 1, definedon =compartment.domain, dirichlet_bbnd = self.cn_bboundary)
        self.scalarspace_navier_bbnd = H1(model.parentmesh, order = 1, definedon =compartment.domain, dirichlet_bbnd = self.navier_bbnd)
        self.discscalarspace = SurfaceL2(model.parentmesh, order = 0, definedon =compartment.domain)

        self.fes = self.scalarspace_bbnd*self.scalarspace_navier_bbnd
        self.gfu = GridFunction(self.fes)
        self.V_h, self.kappa_h = self.gfu.components
        
        self.output_fields["velocity"] = OutputField(self.V_h, "velocity", BND)
        self.output_fields["mean_curvature"] = OutputField(self.kappa_h, "mean_curvature", BND)

        dim = model.dim
        self.ns = specialcf.normal(dim)
        self.Ps = Id(dim) - OuterProduct(self.ns, self.ns)
        if model.dim == 2:
            self.identity = CF((x,y))
        else:
            self.identity = CF((x,y,z))

        self.pre_normal = GridFunction(self.vectorspace)
        self.normal = GridFunction(self.vectorspace)
        self.Amap_h = GridFunction(self.vectorspace_bbnd)
        self.kappa_h_old = GridFunction(self.scalarspace_navier_bbnd)
        self.W_h_old = GridFunction(self.discscalarspace)
        self.J_h_old = GridFunction(self.discscalarspace)

        self.sp_curv_gfu = GridFunction(self.scalarspace_bbnd)

    def Initialize(self):

        deform = self.model.dX

        fes0 = self.vectorspace_bbnd*self.scalarspace_bbnd
        (dY0, kappa0), (nu0, zeta0) = fes0.TnT()
        gfu0 = GridFunction(fes0)
        dY0_h, kappa0_h = gfu0.components

        ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])

        A0 = BilinearForm(fes0)
        A0 += InnerProduct(dY0*self.ns, zeta0)*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = deform)
        A0 += InnerProduct(kappa0*self.ns, nu0)*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = deform)
        A0 += InnerProduct(grad(dY0).Trace(), grad(nu0).Trace())*ds(deformation = deform)
        A0.Assemble()
        invA0 = A0.mat.Inverse(freedofs = fes0.FreeDofs())

        F0 = LinearForm(fes0)
        F0 += -InnerProduct(self.Ps, grad(nu0).Trace())*ds(deformation = deform)
        F0.Assemble()

        gfu0.vec.data = invA0*F0.vec
        self.kappa_h.vec.data = kappa0_h.vec.data

        if self.model.io.root != None:
            output_vtk_folder = os.path.join(self.model.io.root, self.name)
            output_vtk_name = os.path.join(output_vtk_folder, self.name)
            os.makedirs(output_vtk_folder, exist_ok=True)
            self.vtk = VTKOutput(self.model.parentmesh,
                                coefs=[self.V_h, self.kappa_h],
                                names =['velocity', 'mean_curvature'],
                                filename= output_vtk_name, 
                                subdivision = self.params['subdivision'])
            #######################
            # TO be corrected for printing of lines too

    def PreProcess(self):

        self.kappa_h_old.vec.data = self.kappa_h.vec.data

        deform = self.model.dX
        dt = self.model.dt.Get()
        vectorV_h_old = self.compartment.ale.ale_vel
        rhs = self.params['rhs']()
        alpha = self.params['alpha']
        beta = self.params['beta']
        gamma = self.params['gamma']

        self.pre_normal.Set(self.ns, dual = True, definedon=self.compartment.domain)
        self.normal.Set(Normalize(self.pre_normal), dual = True, definedon =self.compartment.domain)
        self.W_h_old.Set(Norm(grad(self.normal).Trace())**2, definedon =self.compartment.domain)
        self.Amap_h.Set(self.identity - dt*vectorV_h_old, dual = True, definedon =self.compartment.domain)
        self.J_h_old.Set(sqrt(Det(Grad(self.Amap_h).Trace().trans*Grad(self.Amap_h).Trace() + OuterProduct(self.ns, self.ns))), definedon=self.compartment.domain)

        (V, kappa), (phi, xsi) = self.fes.TnT()
        
        self.A = BilinearForm(self.fes)
        self.A += InnerProduct(V, phi)*ds(deformation = deform)
        self.A += -alpha*InnerProduct(grad(kappa).Trace(), grad(phi).Trace())*ds(deformation = deform)
        self.A += alpha*InnerProduct(self.W_h_old*kappa, phi)*ds(deformation = deform)
        self.A += -alpha*0.5*InnerProduct((self.kappa_h_old-self.params['sp_curv'])*self.kappa_h_old*kappa, phi)*ds(deformation = deform)
        self.A += -beta*InnerProduct(grad(kappa).Trace(), grad(phi).Trace())*ds(deformation = deform)
        self.A += -gamma*InnerProduct(kappa, phi)*ds(deformation = deform)

        self.A += InnerProduct(kappa/dt,xsi)*ds(deformation = deform)
        self.A += -0.5*(InnerProduct(vectorV_h_old, grad(kappa).Trace()*xsi) - InnerProduct(vectorV_h_old, grad(xsi).Trace()*kappa))*ds(deformation = deform)
        self.A += InnerProduct(grad(V).Trace(), grad(xsi).Trace())*ds(deformation = deform)
        self.A += -InnerProduct(self.W_h_old*V, xsi)*ds(deformation = deform)
        self.A += 0.5*InnerProduct(V, (self.kappa_h_old - self.params['sp_curv'])*self.kappa_h_old*xsi)*ds(deformation = deform)

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = self.fes.FreeDofs())

        self.F = LinearForm(self.fes)

        self.F += InnerProduct(rhs,phi)*ds(deformation = deform)

        self.F += alpha*InnerProduct(self.W_h_old*self.params['sp_curv'], phi)*ds(deformation = deform)
        self.F += -alpha*0.5*InnerProduct((self.kappa_h_old-self.params['sp_curv'])*self.kappa_h_old*self.params['sp_curv'], phi)*ds(deformation = deform)

        self.F += InnerProduct(self.params['sp_curv']/dt,xsi)*ds(deformation = deform)
        self.F += InnerProduct((self.kappa_h_old - self.params['sp_curv'])/dt*sqrt(self.J_h_old),xsi)*ds(deformation = deform)
        self.F += -0.5*(- InnerProduct(vectorV_h_old, grad(xsi).Trace()*self.params['sp_curv']))*ds(deformation = deform)

        self.F.Assemble()

    def Solve(self):

        self.kappa_h.Set(self.params['sp_curv'], definedon = self.model.parentmesh.BBoundaries(self.navier_bbnd))
        self.V_h.vec.data[:] = 0

        self.A.Assemble()
        self.invA.Update()
        self.F.Assemble()

        self.gfu.vec.data += self.invA*self.F.vec

    def PostProcess(self):

        pass