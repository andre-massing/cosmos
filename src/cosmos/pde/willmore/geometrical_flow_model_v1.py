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

    def __init__(self, name:str = 'GeometricalFlowModel', model:CosmosModel = None, compartment:CosmosCompartment = None):
        
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
        self.params["area_preserving"] = False
        self.params["volume_preserving"] = False
        self.params["kappa0"] = None

        self.vectorspace_bbnd = VectorH1(model.parentmesh, order = 1, definedon = compartment.domain, dirichlet_bbnd = self.cn_bboundary)
        self.vectorspace = VectorH1(model.parentmesh, order = 1, definedon =compartment.domain)
        self.scalarspace_bbnd = H1(model.parentmesh, order = 1, definedon =compartment.domain, dirichlet_bbnd = self.cn_bboundary)
        self.scalarspace_navier_bbnd = H1(model.parentmesh, order = 1, definedon =compartment.domain, dirichlet_bbnd = self.navier_bbnd)
        self.discscalarspace = SurfaceL2(model.parentmesh, order = 0, definedon =compartment.domain)

        self.fes = self.scalarspace_bbnd*self.scalarspace_navier_bbnd
        self.gfu = GridFunction(self.fes)
        self.V_h, self.kappa_h = self.gfu.components

        dim = model.dim
        self.ns = specialcf.normal(dim)
        self.Ps = Id(dim) - OuterProduct(self.ns, self.ns)
        if model.dim == 2:
            self.identity = CF((x,y))
        else:
            self.identity = CF((x,y,z))

        self.pre_normal = GridFunction(self.vectorspace)
        self.normal = GridFunction(self.vectorspace)
        self.kappa_h_old = GridFunction(self.scalarspace_navier_bbnd)
        self.W_h_old = GridFunction(self.discscalarspace)

        self.sp_curv_gfu = GridFunction(self.scalarspace_bbnd)

        self.lam = GridFunction(NumberSpace(model.parentmesh, definedon=compartment.domain))
        self.mu = GridFunction(NumberSpace(model.parentmesh, definedon=compartment.domain))
        self.one = GridFunction(NumberSpace(model.parentmesh, definedon=compartment.domain))
        self.one.Set(1, definedon=compartment.domain)

        if self.model.dim == 2:
            self.vtk_gfu = [GridFunction(H1(self.model.parentmesh, order = 1)) for i in range(2)]
        else:
            self.vtk_gfu = self.gfu.components
        self.vtk_names =[self.name + '_V', self.name + '_kappa']

    def Initialize(self):

        if self.params['kappa0']:
            self.kappa_h.Set(self.params['kappa0'], dual = True, definedon = self.compartment.domain)
        else:
            fes0 = self.vectorspace_bbnd*self.scalarspace_bbnd
            (dY0, kappa0), (nu0, zeta0) = fes0.TnT()
            gfu0 = GridFunction(fes0)
            dY0_h, kappa0_h = gfu0.components

            ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])

            A0 = BilinearForm(fes0)
            A0 += InnerProduct(dY0*self.ns, zeta0)*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = self.model.ale.Yo)
            A0 += InnerProduct(kappa0*self.ns, nu0)*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = self.model.ale.Yo)
            A0 += InnerProduct(grad(dY0).Trace(), grad(nu0).Trace())*ds(deformation = self.model.ale.Yo)
            A0.Assemble()
            invA0 = A0.mat.Inverse(freedofs = fes0.FreeDofs())

            F0 = LinearForm(fes0)
            F0 += -InnerProduct(self.Ps, grad(nu0).Trace())*ds(deformation = self.model.ale.Yo)
            F0.Assemble()

            gfu0.vec.data = invA0*F0.vec
            self.kappa_h.vec.data = kappa0_h.vec.data

        if self.model.dim == 2:
            if self.params['printing']:
                for i, gfu in enumerate(self.vtk_gfu):
                    gfu.Set(self.gfu.components[i], definedon = self.compartment.domain)

    def PreProcess(self):

        self.kappa_h_old.vec.data = self.kappa_h.vec.data
        self.pre_normal.Set(self.ns, dual = True, definedon=self.compartment.domain)
        self.normal.Set(Normalize(self.pre_normal), dual = True, definedon =self.compartment.domain)
        self.W_h_old.Set(Norm(grad(self.normal).Trace())**2, definedon =self.compartment.domain)

    def Solve(self):

        rhs = self.params['rhs']()
        alpha = self.params['alpha']
        beta = self.params['beta']
        gamma = self.params['gamma']

        (V, kappa), (phi, xsi) = self.fes.TnT()
        
        self.A = BilinearForm(self.fes)
        self.A += InnerProduct(V, phi)*ds(deformation = self.model.ale.Yo)
        self.A += -alpha*InnerProduct(grad(kappa).Trace(), grad(phi).Trace())*ds(deformation = self.model.ale.Yo)
        self.A += alpha*InnerProduct(self.W_h_old*kappa, phi)*ds(deformation = self.model.ale.Yo)
        self.A += -alpha*0.5*InnerProduct((self.kappa_h_old-self.params['sp_curv'])*self.kappa_h_old*kappa, phi)*ds(deformation = self.model.ale.Yo)
        self.A += -beta*InnerProduct(grad(kappa).Trace(), grad(phi).Trace())*ds(deformation = self.model.ale.Yo)
        self.A += -gamma*InnerProduct(kappa, phi)*ds(deformation = self.model.ale.Yo)

        self.A += InnerProduct(kappa/self.model.dt,xsi)*ds(deformation = self.model.ale.Yo)
        self.A += -0.5*(InnerProduct(self.model.ale.Wo, grad(kappa).Trace()*xsi) - InnerProduct(self.model.ale.Wo, grad(xsi).Trace()*kappa))*ds(deformation = self.model.ale.Yo)
        self.A += InnerProduct(grad(V).Trace(), grad(xsi).Trace())*ds(deformation = self.model.ale.Yo)
        self.A += -InnerProduct(self.W_h_old*V, xsi)*ds(deformation = self.model.ale.Yo)
        self.A += 0.5*InnerProduct(V, (self.kappa_h_old - self.params['sp_curv'])*self.kappa_h_old*xsi)*ds(deformation = self.model.ale.Yo)

        self.A += -1*InnerProduct(self.lam*kappa, phi)*ds(deformation = self.model.ale.Yo)

        #### Implict coupling to geometry!
        self.A += 0.5*InnerProduct(InnerProduct(Grad(self.model.ale.X).Trace(), Grad(self.model.ale.W).Trace())*kappa, phi)*ds(deformation = self.model.ale.Yo)

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = self.fes.FreeDofs())

        self.F = LinearForm(self.fes)

        self.F += InnerProduct(rhs,phi)*ds(deformation = self.model.ale.Yo)

        self.F += alpha*InnerProduct(self.W_h_old*self.params['sp_curv'], phi)*ds(deformation = self.model.ale.Yo)
        self.F += -alpha*0.5*InnerProduct((self.kappa_h_old-self.params['sp_curv'])*self.kappa_h_old*self.params['sp_curv'], phi)*ds(deformation = self.model.ale.Yo)

        self.F += InnerProduct(self.params['sp_curv']/self.model.dt,xsi)*ds(deformation = self.model.ale.Yo)
        self.F += InnerProduct((self.kappa_h_old - self.params['sp_curv'])/self.model.dt,xsi)*ds(deformation = self.model.ale.Yo)
        self.F += -0.5*(- InnerProduct(self.model.ale.Wo, grad(xsi).Trace()*self.params['sp_curv']))*ds(deformation = self.model.ale.Yo)

        self.F += InnerProduct(self.mu, phi)*ds(deformation = self.model.ale.Yo)

        #### Implict coupling to geometry!
        self.F += 0.5*InnerProduct(InnerProduct(Grad(self.model.ale.X).Trace(), Grad(self.model.ale.W).Trace())*self.params['sp_curv'], phi)*ds(deformation = self.model.ale.Yo)

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

                self.kappa_h.Set(self.params['sp_curv'], definedon = self.model.parentmesh.BBoundaries(self.navier_bbnd))
                self.V_h.vec.data[:] = 0

                if self.params['area_preserving'] and not self.params['volume_preserving']:
                    
                    self.A.Assemble()
                    self.invA.Update()
                    self.F.Assemble()
                    res = self.A.mat*self.gfu.vec
                    self.gfu.vec.data += self.invA*(self.F.vec-res)

                    lam_new = Integrate((-self.V_h + self.lam*self.kappa_h)*self.kappa_h, mesh = self.model.parentmesh, VOL_or_BND = BND)/Integrate(self.kappa_h**2, mesh = self.model.parentmesh, VOL_or_BND = BND)

                elif self.params['volume_preserving'] and not self.params['area_preserving']:

                    self.F.Assemble()
                    res = self.A.mat*self.gfu.vec
                    self.gfu.vec.data += self.invA*(self.F.vec-res)

                    mu_new = Integrate(-self.V_h + self.mu, mesh = self.model.parentmesh, VOL_or_BND = BND)/Integrate(self.one, mesh = self.model.parentmesh, VOL_or_BND = BND)

                elif self.params['volume_preserving'] and self.params['area_preserving']:

                    self.A.Assemble()
                    self.invA.Update()
                    self.F.Assemble()
                    res = self.A.mat*self.gfu.vec
                    self.gfu.vec.data += self.invA*(self.F.vec-res)

                    M = np.zeros((2,2))
                    c = np.zeros(2)
                    M[0, 0] = Integrate(self.kappa_h**2, mesh = self.model.parentmesh, VOL_or_BND = BND)
                    M[0, 1] = Integrate(self.kappa_h, mesh = self.model.parentmesh, VOL_or_BND = BND)
                    M[1, 0] = Integrate(self.kappa_h, mesh = self.model.parentmesh, VOL_or_BND = BND)
                    M[1, 1] = Integrate(self.one, mesh = self.model.parentmesh, VOL_or_BND = BND)
                    c[0] = Integrate((-self.V_h + self.lam*self.kappa_h + self.mu)*self.kappa_h, mesh = self.model.parentmesh, VOL_or_BND = BND)
                    c[1] = Integrate(-self.V_h + self.lam*self.kappa_h + self.mu, mesh = self.model.parentmesh, VOL_or_BND = BND)
                    x = np.linalg.solve(M, c)
                    lam_new = x[0]
                    mu_new = x[1]

                err_lam = abs(lam_new-lam_old)
                err_mu = abs(mu_new-mu_old)
                if err_lam<tol and err_mu<tol:
                    # print('Converged in ', iter, ' iterations')
                    break

            if iter == max_iter:
                raise Exception('Number of iterations for internal solver exceeded')

        else:

            self.kappa_h.Set(self.params['sp_curv'], definedon = self.model.parentmesh.BBoundaries(self.navier_bbnd))
            self.V_h.vec.data[:] = 0

            self.A.Assemble()
            self.invA.Update()
            self.F.Assemble()

            res = self.A.mat*self.gfu.vec
            self.gfu.vec.data += self.invA*(self.F.vec-res)

    def PostProcess(self):

        if self.model.dim == 2:
            if self.params['printing']:
                for i, gfu in enumerate(self.vtk_gfu):
                    gfu.Set(self.gfu.components[i], definedon = self.compartment.domain)

        del self.A
        del self.invA
        del self.F

def MC1(mesh):
    order_g = mesh.GetCurveOrder()
    V = VectorH1(mesh, order=order_g)
    dV = VectorFacetSurface(mesh, order=order_g)
    W = V*dV
    (kappa, dkappa), (eta, deta) = W.TnT()
    
    Idh = GridFunction(V)
    Idh.Set(CF((x,y,z)), definedon=mesh.Boundaries(".*"))
    
    # Compute mesh size, normal and tangential vectors 
    h = specialcf.mesh_size
    ns = specialcf.normal(mesh.dim)
    tE = specialcf.tangential(mesh.dim)
    nE = Cross(ns, tE)
    
    # Define bilinear form 
    m = BilinearForm(W)
    m += InnerProduct(kappa.Trace(), eta.Trace())*ds
    jump_dkappadn = (kappa.Trace().Deriv()*nE-dkappa.Trace())
    jump_detadn = (eta.Trace().Deriv()*nE-deta.Trace())
    # jump_dkappadn = (kappa.Trace().Deriv().trans*nE-dkappa.Trace())
    # jump_detadn = (eta.Trace().Deriv().trans*nE-deta.Trace())
    gamma_E = 0.001
    m += gamma_E*h*InnerProduct(jump_dkappadn,jump_detadn)*ds(element_boundary=True)
    
    l = LinearForm(W)
    # TODO: Determine right sign for normal vector
    l += -1*InnerProduct(grad(Idh).Trace(), grad(eta).Trace())*ds
    
    # Assemble and solve system
    m.Assemble()
    # print(f"Norm(m) = {m.mat.AsVector().Norm()}")
    l.Assemble()
    Minv = m.mat.Inverse(W.FreeDofs(), inverse="umfpack")
    
    wh = GridFunction(W)
    wh.vec.data = Minv * l.vec
    kappah, _ = wh.components
        
    return kappah

def MC2(mesh):
    n = specialcf.normal(3)
    t = specialcf.tangential(3)
    mu = Cross(n,t)

    order = mesh.GetCurveOrder()
    
    # Average normal vector
    gfF = GridFunction(VectorFacetSurface(mesh,order=order-1))
    gfF.Set(n, dual=True, definedon=mesh.Boundaries(".*"))
    
    fes = HDivDivSurface(mesh,order=order-1)
    sigma,tau = fes.TnT()
    sigma,tau = sigma.Trace(),tau.Trace()
    
    a = BilinearForm(fes, symmetric=True)
    a += InnerProduct(sigma,tau)*ds
    
    # Grad(n) = specialcf.Weingarten(3)
    f = LinearForm(fes)
    f += InnerProduct(Grad(n),tau)*ds \
            + (pi/2-acos(Normalize(gfF)*mu))*tau*mu*mu*ds(element_boundary=True)
    
    gflift = GridFunction(fes)
    
    with TaskManager():
        a.Assemble()
        f.Assemble()
        gflift.vec.data = -1*a.mat.Inverse(fes.FreeDofs(),inverse="sparsecholesky")*f.vec
        
    return gflift