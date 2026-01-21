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

class GeometricalFlowStationaryModel(BasePDEModel):

    def __init__(self, name:str = 'GeometricalFlowStationaryModel', model:CosmosModel = None, compartment:CosmosCompartment = None):
        
        super().__init__(name=name, model = model, compartment=compartment)
        
        self.is_bnd = True
        self.is_vol = False
        self.name = name
        self.model = model
        self.compartment = compartment

        self.cn_bboundary = compartment.clamped_bbnd + '|' + compartment.navier_bbnd
        self.navier_bbnd = compartment.navier_bbnd
        self.params["subdivision"] = 0
        self.params["rhs"] = Field(CF(0))
        self.params["alpha"] = CF(1)
        self.params["beta"] = CF(0)
        self.params["gamma"] = CF(0)
        self.params["area_preserving"] = False
        self.params["volume_preserving"] = False

        self.vectorspace_bbnd = VectorH1(model.parentmesh, order = 1, definedon = compartment.domain, dirichlet_bbnd = self.cn_bboundary)
        self.vectorspace = VectorH1(model.parentmesh, order = 1, definedon =compartment.domain)
        self.scalarspace_bbnd = H1(model.parentmesh, order = 1, definedon =compartment.domain, dirichlet_bbnd = self.cn_bboundary)
        self.scalarspace_navier_bbnd = H1(model.parentmesh, order = 1, definedon =compartment.domain, dirichlet_bbnd = self.navier_bbnd)
        self.discscalarspace = SurfaceL2(model.parentmesh, order = 0, definedon =compartment.domain)

        self.fes = self.scalarspace_bbnd*self.scalarspace_navier_bbnd*self.scalarspace_navier_bbnd
        self.gfu = GridFunction(self.fes)
        self.V_h, self.kappa_h, self.sp_curv_h = self.gfu.components
        
        self.output_fields["velocity"] = OutputField(self.V_h, "velocity", BND)
        self.output_fields["mean_curvature"] = OutputField(self.kappa_h, "mean_curvature", BND)
        self.output_fields["spontaneous_curvature"] = OutputField(self.sp_curv_h, "spontaneous_curvature", BND)

        dim = model.dim
        self.ns = specialcf.normal(dim)
        self.Ps = Id(dim) - OuterProduct(self.ns, self.ns)
        if model.dim == 2:
            self.identity = CF((x,y))
        else:
            self.identity = CF((x,y,z))
        tE = specialcf.tangential(self.model.dim)
        if self.model.dim == 2:
            self.nE = tE
        else:
            self.nE = Cross(self.ns, tE)

        self.pre_normal = GridFunction(self.vectorspace)
        self.normal = GridFunction(self.vectorspace)
        self.Amap_h = GridFunction(self.vectorspace_bbnd)
        self.kappa_h_old = GridFunction(self.scalarspace_navier_bbnd)
        self.sp_curv_h_old = GridFunction(self.scalarspace_navier_bbnd)
        self.W_h_old = GridFunction(self.discscalarspace)
        self.J_h_old = GridFunction(self.discscalarspace)

        self.lam = GridFunction(NumberSpace(model.parentmesh, definedon=compartment.domain))
        self.mu = GridFunction(NumberSpace(model.parentmesh, definedon=compartment.domain))
        self.one = GridFunction(NumberSpace(model.parentmesh, definedon=compartment.domain))
        self.one.Set(1, definedon=compartment.domain)

    def Initialize(self):

        deform = self.model.dX

        ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])

        kappa_mc, eta_mc = self.vectorspace.TnT()
        self.A_mc = BilinearForm(self.vectorspace, symmetric = True)
        self.A_mc += (kappa_mc*eta_mc)*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = deform)
        self.A_mc.Assemble()
        self.invA_mc = self.A_mc.mat.Inverse(freedofs = self.vectorspace.FreeDofs())
        self.F_mc = LinearForm(self.vectorspace)
        self.F_mc += -InnerProduct(self.Ps, grad(eta_mc).Trace())*ds(deformation = deform)
        if self.model.parentmesh.dim == 2:
            gfBB = GridFunction(H1(self.model.parentmesh, order=1, definedon=self.compartment.domain))
            gfBB.Set(1, definedon=self.model.parentmesh.BBoundaries(self.cn_bboundary))
            self.F_mc += InnerProduct(self.nE, eta_mc)*gfBB*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = deform, element_boundary = True)
        elif self.model.parentmesh.dim == 3:
            gfBB = GridFunction(FacetSurface(self.model.parentmesh, order=0, definedon = self.compartment.domain))
            gfBB.Set(1, definedon=self.model.parentmesh.BBoundaries(self.cn_bboundary))
            self.F_mc += InnerProduct(self.nE, eta_mc)*gfBB*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = deform, element_boundary = True)
        self.F_mc.Assemble()
        
        gfu0 = GridFunction(self.vectorspace)

        gfu0.vec.data = self.invA_mc*self.F_mc.vec
        self.kappa_h.Set(gfu0*self.ns, definedon = self.compartment.domain)
        self.sp_curv_h.Set(gfu0*self.ns, definedon = self.compartment.domain)

        if self.params['printing']:
            output_vtk_folder = os.path.join(self.model.io.root, self.name)
            output_vtk_name = os.path.join(output_vtk_folder, self.name)
            os.makedirs(output_vtk_folder, exist_ok=True)
            if self.model.dim == 2:
                self.gfu_vtk = [GridFunction(H1(self.model.parentmesh, order = 1)) for i in range(3)]
                gfu_one = GridFunction(H1(self.model.parentmesh, order = 1))
                gfu_one.Set(1, definedon = self.compartment.domain)
            else:
                self.gfu_vtk = self.gfu.components
                gfu_one = GridFunction(H1(self.model.parentmesh, order = 1, definedon = self.compartment.domain))
                gfu_one.Set(1, definedon = self.compartment.domain)
            self.vtk = VTKOutput(self.model.parentmesh,
                                coefs=[self.gfu_vtk[i] for i in range(3)]+ [gfu_one],
                                names =['velocity', 'mean_curvature', 'spontaneous_curvature', 'indicator'],
                                filename= output_vtk_name, 
                                subdivision = self.params['subdivision'])
    def PreProcess(self):

        self.kappa_h_old.vec.data = self.kappa_h.vec.data
        self.sp_curv_h_old.vec.data = self.sp_curv_h.vec.data

        self.pre_normal.Set(self.ns, dual = True, definedon=self.compartment.domain)
        self.normal.Set(Normalize(self.pre_normal), dual = True, definedon =self.compartment.domain)
        self.W_h_old.Set(Norm(grad(self.normal).Trace())**2, definedon =self.compartment.domain)
        self.Amap_h.Set(self.identity - self.model.dt*self.model.ale.ale_velocity, dual = True, definedon =self.compartment.domain)
        self.J_h_old.Set(sqrt(Det(Grad(self.Amap_h).Trace().trans*Grad(self.Amap_h).Trace() + OuterProduct(self.ns, self.ns))), definedon=self.compartment.domain)

    def Solve(self):

        deform = self.model.dX
        rhs = self.params['rhs']()
        alpha = self.params['alpha']
        beta = self.params['beta']
        gamma = self.params['gamma']

        (V, kappa, sp_curv), (phi, xsi, zeta) = self.fes.TnT()
        
        self.A = BilinearForm(self.fes)
        self.A += InnerProduct(V, phi)*ds(deformation = deform)
        self.A += -alpha*InnerProduct(grad(kappa).Trace(), grad(phi).Trace())*ds(deformation = deform)
        self.A += alpha*InnerProduct(grad(sp_curv).Trace(), grad(phi).Trace())*ds(deformation = deform)
        self.A += alpha*InnerProduct(self.W_h_old*kappa, phi)*ds(deformation = deform)
        self.A += -alpha*InnerProduct(self.W_h_old*sp_curv, phi)*ds(deformation = deform)
        self.A += -alpha*0.5*InnerProduct((self.kappa_h_old-self.sp_curv_h_old)*self.kappa_h_old*kappa, phi)*ds(deformation = deform)
        self.A += alpha*0.5*InnerProduct((self.kappa_h_old-self.sp_curv_h_old)*self.kappa_h_old*sp_curv, phi)*ds(deformation = deform)
        self.A += -beta*InnerProduct(grad(kappa).Trace(), grad(phi).Trace())*ds(deformation = deform)
        self.A += -gamma*InnerProduct(kappa, phi)*ds(deformation = deform)

        self.A += InnerProduct(kappa/self.model.dt,xsi)*ds(deformation = deform)
        self.A += -1*InnerProduct(sp_curv/self.model.dt,xsi)*ds(deformation = deform)
        self.A += -0.5*(InnerProduct(self.model.ale.ale_velocity, grad(kappa).Trace()*xsi) - InnerProduct(self.model.ale.ale_velocity, grad(xsi).Trace()*kappa))*ds(deformation = deform)
        self.A += 0.5*(InnerProduct(self.model.ale.ale_velocity, grad(sp_curv).Trace()*xsi) - InnerProduct(self.model.ale.ale_velocity, grad(xsi).Trace()*sp_curv))*ds(deformation = deform)
        self.A += InnerProduct(grad(V).Trace(), grad(xsi).Trace())*ds(deformation = deform)
        self.A += -InnerProduct(self.W_h_old*V, xsi)*ds(deformation = deform)
        self.A += 0.5*InnerProduct(V, (self.kappa_h_old - self.sp_curv_h_old)*self.kappa_h_old*xsi)*ds(deformation = deform)

        self.A += -1*InnerProduct(self.lam*kappa, phi)*ds(deformation = deform)

        self.A += InnerProduct(sp_curv/self.model.dt, zeta)*ds(deformation = deform)
        self.A += InnerProduct(-self.model.ale.ale_velocity*grad(sp_curv).Trace(), zeta)*ds(deformation = deform)

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = self.fes.FreeDofs())

        self.F = LinearForm(self.fes)

        self.F += InnerProduct(rhs,phi)*ds(deformation = deform)

        self.F += InnerProduct((self.kappa_h_old - self.sp_curv_h_old)/self.model.dt*sqrt(self.J_h_old),xsi)*ds(deformation = deform)
        self.F += InnerProduct(self.sp_curv_h_old/self.model.dt, zeta)*ds(deformation = deform)

        self.F += InnerProduct(self.mu, phi)*ds(deformation = deform)

        if self.params['area_preserving'] or self.params['volume_preserving']:

            iter = 0
            lam_old = 0
            lam_new = 0
            mu_old = 0
            mu_new = 0
            tol = 1e-6

            max_iter = 10

            while iter<max_iter:

                lam_old = lam_new
                mu_old = mu_new
                self.lam.vec[:] = lam_old
                self.mu.vec[:] = mu_old

                iter += 1

                self.kappa_h.Set(self.sp_curv_h_old, definedon = self.model.parentmesh.BBoundaries(self.navier_bbnd))
                self.sp_curv_h.Set(self.sp_curv_h_old, definedon = self.model.parentmesh.BBoundaries(self.navier_bbnd))
                self.V_h.vec.data[:] = 0

                self.A.Assemble()
                self.invA.Update()
                self.F.Assemble()

                res = self.A.mat*self.gfu.vec
                self.gfu.vec.data += self.invA*(self.F.vec-res)

                if self.params['area_preserving'] and not self.params['volume_preserving']:
                    lam_new = Integrate((-self.V_h + self.lam*self.kappa_h)*self.kappa_h, mesh = self.model.parentmesh, VOL_or_BND = BND)/Integrate(self.kappa_h**2, mesh = self.model.parentmesh, VOL_or_BND = BND)
                elif self.params['volume_preserving'] and not self.params['area_preserving']:
                    mu_new = Integrate(-self.V_h + self.mu, mesh = self.model.parentmesh, VOL_or_BND = BND)/Integrate(self.one, mesh = self.model.parentmesh, VOL_or_BND = BND)
                elif self.params['volume_preserving'] and self.params['area_preserving']:
                    M = np.zeros((2,2))
                    c = np.zeros(2)
                    M[0, 0] = Integrate(self.kappa_h**2, mesh = self.model.parentmesh, VOL_or_BND = BND)
                    M[0, 1] = Integrate(self.kappa_h, mesh = self.model.parentmesh, VOL_or_BND = BND)
                    M[1, 0] = M[0, 1]
                    M[1, 1] = Integrate(self.one, mesh = self.model.parentmesh, VOL_or_BND = BND)
                    c[0] = Integrate((-self.V_h + self.lam*self.kappa_h + self.mu)*self.kappa_h, mesh = self.model.parentmesh, VOL_or_BND = BND)
                    c[1] = Integrate(-self.V_h + self.lam*self.kappa_h + self.mu, mesh = self.model.parentmesh, VOL_or_BND = BND)
                    x = np.linalg.solve(M, c)
                    lam_new = x[0]
                    mu_new = x[1]

                err_lam = abs(lam_new-lam_old)
                err_mu = abs(mu_new-mu_old)
                print(err_lam, err_mu)
                if err_lam<tol and err_mu<tol:
                    # print('Converged in ', iter, ' iterations')
                    break

            if iter == max_iter:
                raise Exception('Number of iterations for internal solver exceeded')
            
        else:

            self.kappa_h.Set(self.sp_curv_h_old, definedon = self.model.parentmesh.BBoundaries(self.navier_bbnd))
            self.sp_curv_h.Set(self.sp_curv_h_old, definedon = self.model.parentmesh.BBoundaries(self.navier_bbnd))
            self.V_h.vec.data[:] = 0

            self.A.Assemble()
            self.invA = self.A.mat.Inverse(freedofs = self.fes.FreeDofs())
            self.F.Assemble()

            res = self.A.mat*self.gfu.vec
            self.gfu.vec.data += self.invA*(self.F.vec-res)

    def PostProcess(self):

        if self.model.dim == 2:
            for i, gfu in enumerate(self.gfu_vtk):
                gfu.Set(self.gfu.components[i], definedon = self.compartment.domain)