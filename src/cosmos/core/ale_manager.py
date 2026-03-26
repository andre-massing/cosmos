import logging
logger = logging.getLogger(__name__)

import numpy as np
import time
from ngsolve import *
from cosmos.config.parameters import get_config
from cosmos.core.field import Field
from cosmos.core.compartment import CosmosCompartment
from ngsolve.webgui import Draw
from ngsolve.solvers import Newton
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cosmos.core.model import CosmosModel

class CosmosALEManager:

    def __init__(self, model:"CosmosModel", kwargs):

        self.params = kwargs
        if 'volume_ALE' in kwargs:
            self.volume_ALE = kwargs['volume_ALE']
        else:
            self.volume_ALE = 'laplace'
        if 'surface_ALE' in kwargs:
            self.surface_ALE = kwargs['surface_ALE']
        else:
            self.surface_ALE = 'mdr'
        self.redistribute = False
        self.ale_elapsed_time = None

        if model.is_bnd:
            self.domain = model.parentmesh.Boundaries('.*')
            self.fes = VectorH1(model.parentmesh, order = model.geo_order,
                                        definedon = self.domain)
            gfu = GridFunction(self.fes)
            model.parentmesh.SetDeformation(gfu)
        else:
            self.domain = model.parentmesh.Materials('.*')
            self.fes = VectorH1(model.parentmesh, order = model.geo_order, definedon = self.domain)
            gfu = GridFunction(self.fes)
            model.parentmesh.SetDeformation(gfu)
        
        self.dY = GridFunction(self.fes)
        self.Y = GridFunction(self.fes)
        self.Yo = GridFunction(self.fes)
        self.Xo = GridFunction(self.fes)
        self.X = GridFunction(self.fes)
        if model.dim == 2:
            self.Xo.Set(CF((x, y)), definedon = self.domain)
            self.X.Set(CF((x, y)), definedon = self.domain)
        elif model.dim == 3:
            self.Xo.Set(CF((x, y, z)), definedon = self.domain)
            self.X.Set(CF((x, y, z)), definedon = self.domain)
        self.W = GridFunction(self.fes)
        self.Wo = GridFunction(self.fes)
        self.gfu_bnd_ale = GridFunction(self.fes)
        self.gfu_bnd_mat = GridFunction(self.fes)
        self.gfu_vol_ale = GridFunction(self.fes)
        self.gfu_vol_mat = GridFunction(self.fes)
        self.dY_mat = GridFunction(self.fes)
        self.V = GridFunction(self.fes)
        self.Vo = GridFunction(self.fes)

        if 'redistribute' in  self.params.keys():
            if isinstance(self.params['redistribute'], bool):
                self.redistribute = self.params['redistribute']
            else:
                raise Exception(f'Mesh redistribution for for model {model.name} must be either True or False')      
        
        if model.is_vol:

            if self.volume_ALE == 'laplace':
                self.V0 = VectorH1(model.parentmesh, order = model.geo_order, dirichlet = model.parentmesh.Boundaries('.*'))
                u, v = self.V0.TnT()
                self.A = BilinearForm(self.V0, symmetric = True)
                self.A += InnerProduct(Grad(u), Grad(v))*dx(deformation = self.Yo)
                self.A.Assemble()
                self.invA = self.A.mat.Inverse(freedofs = self.V0.FreeDofs())

            elif self.volume_ALE == 'linel':
                self.V0 = VectorH1(model.parentmesh, order = model.geo_order, dirichlet = model.parentmesh.Boundaries('.*'))
                u, v = self.V0.TnT()
                self.A = BilinearForm(self.V0, symmetric = True)
                h = specialcf.mesh_size
                E, nu = 1/h**model.dim, 0.49
                mu  = E / 2 / (1+nu)
                lam = E * nu / ((1+nu)*(1-2*nu))
                def Stress(strain):
                    return 2*mu*strain + lam*Trace(strain)*Id(model.dim)    
                self.A += InnerProduct(Stress(Sym(Grad(u))), Sym(Grad(v)))*dx(deformation = self.Yo)
                self.A.Assemble()
                self.invA = self.A.mat.Inverse(freedofs = self.V0.FreeDofs())

            elif self.volume_ALE == 'linel0':
                self.V0 = VectorH1(model.parentmesh, order = model.geo_order, dirichlet = model.parentmesh.Boundaries('.*'))
                u, v = self.V0.TnT()
                self.A = BilinearForm(self.V0, symmetric = True)
                h = specialcf.mesh_size
                E, nu = 1/h**model.dim, 0.49
                mu  = E / 2 / (1+nu)
                lam = E * nu / ((1+nu)*(1-2*nu))
                def Stress(strain):
                    return 2*mu*strain + lam*Trace(strain)*Id(model.dim)    
                self.A += InnerProduct(Stress(Sym(Grad(u))), Sym(Grad(v)))*dx
                self.A.Assemble()
                self.invA = self.A.mat.Inverse(freedofs = self.V0.FreeDofs())

            elif self.volume_ALE == 'nonlinel':
                self.V0 = VectorH1(model.parentmesh, order = model.geo_order, dirichlet = model.parentmesh.Boundaries('.*'))
                u, v = self.V0.TnT()
                self.A = BilinearForm(self.V0, symmetric = True)
                h = specialcf.mesh_size
                E, nu = 1/h, 0.3
                mu  = E / 2 / (1+nu)
                lam = E * nu / ((1+nu)*(1-2*nu))
                I = Id(model.dim)
                F = I + Grad(u)
                C = F.trans * F
                E = 0.5 * (C-I)
                def Pow(a, b):
                    return a**b  # exp (log(a)*b)
                def NeoHooke (C):
                    return 0.5 * mu * (Trace(C-I) + 2*mu/lam * Pow(Det(C),-lam/2/mu) - 1)
                self.A += Variation (NeoHooke(C).Compile()*dx)

            else:
                raise Exception('ALE extension to volume type not known')

        self.vtk_gfu=[self.dY, self.Y, self.X, self.W, 
                    self.V]
        self.vtk_names =['ale_dY', 'ale_Y', 'ale_X',
                'ale_W', 'ale_V']

    def initialize(self, model:"CosmosModel"):

        self.bnd_ales = {}
        self.bnd_mats = {}
        self.vol_ales = {}
        self.vol_mats = {}
        for ale in model.ales:
            if isinstance(ale, CosmosBndALEField):
                self.bnd_ales[ale.compartment.domain_id] = ale.ale_displ
                self.bnd_mats[ale.compartment.domain_id] = ale.mat_displ
            elif isinstance(ale, CosmosVolALEField):
                self.vol_ales[ale.compartment.domain_id] = ale.ale_displ
                self.vol_mats[ale.compartment.domain_id] = ale.mat_displ

        self.bnd_ale_cf = model.parentmesh.BoundaryCF(self.bnd_ales, default = CF((0,)*model.dim))
        self.bnd_mat_cf = model.parentmesh.BoundaryCF(self.bnd_mats, default = CF((0,)*model.dim))
        self.vol_ale_cf = model.parentmesh.MaterialCF(self.vol_ales, default = CF((0,)*model.dim))
        self.vol_mat_cf = model.parentmesh.MaterialCF(self.vol_mats, default = CF((0,)*model.dim))

    def solve_ale(self, model:"CosmosModel"):

        start = time.time()

        self.dY.vec.data[:] = 0
        self.dY_mat.vec.data[:] = 0
        
        for ale in model.ales:
            ale.update(model, self.redistribute)

        if self.bnd_ales:
            self.gfu_bnd_ale.Set(self.bnd_ale_cf, definedon = model.parentmesh.Boundaries('.*'))
            if model.is_vol:
                self._extend_displacement_to_bulk(self.gfu_bnd_ale)
            self.dY.vec.data += self.gfu_bnd_ale.vec.data

            self.gfu_bnd_mat.Set(self.bnd_mat_cf, definedon = model.parentmesh.Boundaries('.*'))
            if model.is_vol:
                self._extend_displacement_to_bulk(self.gfu_bnd_mat)
            self.dY_mat.vec.data += self.gfu_bnd_mat.vec.data

        if self.vol_ales:
            self.gfu_vol_ale.Set(self.vol_ale_cf)
            self.gfu_vol_mat.Set(self.vol_mat_cf)

            self.dY.vec.data += self.gfu_vol_ale.vec.data
            self.dY_mat.vec.data += self.gfu_vol_mat.vec.data

        if self.bnd_ales or self.vol_ales:
            self.Y.vec.data = self.Yo.vec.data + self.dY.vec.data
            self.X.vec.data = self.Xo.vec.data + self.dY.vec.data
            self.W.Set(self.dY/model.dt, definedon = self.domain)
            self.V.Set(self.dY_mat/model.dt, definedon = self.domain)

        stop = time.time()

        self.ale_elapsed_time = stop-start

    def finalize(self, model:"CosmosModel"):

        if self.bnd_ales or self.vol_ales:
            model.parentmesh.deformation.vec.data = self.Y.vec.data
            self.Yo.vec.data = self.Y.vec.data
            self.Xo.vec.data = self.X.vec.data
            self.Wo.vec.data = self.W.vec.data
            self.Vo.vec.data = self.V.vec.data

    def _extend_displacement_to_bulk(self, gfu):

        if self.volume_ALE == 'laplace' or self.volume_ALE == 'linel':

            self.A.Assemble()
            vec = -1*self.A.mat*gfu.vec
            self.invA.Update()
            gfu.vec.data += self.invA*vec

        elif self.volume_ALE == 'linel0':

            gfu_l0 = GridFunction(gfu.space)
            gfu_l0.vec.data += gfu.vec.data + self.Yo.vec.data

            self.A.Assemble()
            vec = -1*self.A.mat*gfu_l0.vec
            self.invA.Update()
            gfu_l0.vec.data += self.invA*vec

            gfu.vec.data = gfu_l0.vec.data - self.Yo.vec.data

        elif self.volume_ALE == 'nonlinel':

            gfu_nl = GridFunction(gfu.space)
            gfu_nl.vec.data += gfu.vec.data + self.Yo.vec.data
            Newton(self.A, gfu_nl, freedofs=self.V0.FreeDofs(), printing = False)
            gfu.vec.data = gfu_nl.vec.data - self.Yo.vec.data

    def reset(self):

        self.Y.vec.data = self.Yo.vec.data
        self.X.vec.data = self.Xo.vec.data
        self.W.vec.data = self.Wo.vec.data
        self.V.vec.data = self.Vo.vec.data

def SimpleNewtonSolve(gfu,a,tol=1e-13,maxits=25):
    res = gfu.vec.CreateVector()
    du = gfu.vec.CreateVector()
    fes = gfu.space
    for it in range(maxits):
        print ("Iteration {:3}  ".format(it),end="")
        a.Apply(gfu.vec, res)
        a.AssembleLinearization(gfu.vec)
        du.data = a.mat.Inverse(fes.FreeDofs()) * res
        gfu.vec.data -= du

        #stopping criteria
        stopcritval = sqrt(abs(InnerProduct(du,res)))
        print ("<A u",it,", A u",it,">_{-1}^0.5 = ", stopcritval)
        if stopcritval < tol:
            break
    if it == maxits-1:
        raise Exception('Maximum number of iterations reached for the Newton Solver')
        
class CosmosBndALEField:

    def __init__(self, name:str, model:"CosmosModel", compartment:CosmosCompartment):

        self.name = name
        self.model = model
        self.compartment = compartment
        self.tangential_velocity = None
        self.normal_velocity = None
        self.domain_velocity = None

        self.ns = specialcf.normal(model.dim)
        self.Ps = Id(model.dim) - OuterProduct(self.ns, self.ns)
        self.Qs = OuterProduct(self.ns, self.ns)

        self.fes = VectorH1(self.model.parentmesh, order = self.model.geo_order, 
                        definedon = self.compartment.domain,
                        dirichlet_bbnd = self.compartment.boundary)
        S = H1(self.model.parentmesh, order = self.model.geo_order, 
                        definedon = self.compartment.domain,
                        dirichlet_bbnd = self.compartment.boundary)
        
        self.ale_displ = GridFunction(self.fes)
        self.mat_displ = GridFunction(self.fes)
        self.ale_vel = GridFunction(self.fes)
        self.mat_vel = GridFunction(self.fes)

        self.gfu_norm_vel = GridFunction(S)

        fes_pp = self.fes*S
        (dX_pp, kappa_pp), (nu_pp, zeta_pp) = fes_pp.TnT()
        self.gfu_pp = GridFunction(fes_pp)

        ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])

        if model.ale.surface_ALE == 'duanli':

            gfu0 = GridFunction(model.ale.Yo.space)

            self.A_pp = BilinearForm(fes_pp, symmetric = True)
            self.A_pp += InnerProduct(dX_pp*self.ns, zeta_pp)*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.ale.Yo)
            self.A_pp += InnerProduct(kappa_pp*self.ns, nu_pp)*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.ale.Yo)
            self.A_pp += InnerProduct(Grad(dX_pp).Trace(), Grad(nu_pp).Trace())*ds(deformation = gfu0)
            self.A_pp.Assemble()
            self.invA_pp = self.A_pp.mat.Inverse(freedofs = fes_pp.FreeDofs())

            self.F_pp = LinearForm(fes_pp)
            self.F_pp += InnerProduct(self.gfu_norm_vel*model.dt, zeta_pp )*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.ale.Yo)
            self.F_pp += -1*InnerProduct(Grad(model.ale.Yo).Trace(), Grad(nu_pp).Trace())*ds(deformation = gfu0)
            self.F_pp += -1*InnerProduct(self.Ps, Grad(nu_pp).Trace())*ds(deformation = gfu0)

        elif model.ale.surface_ALE == 'mdr':

            self.A_pp = BilinearForm(fes_pp)
            self.A_pp += InnerProduct(dX_pp*self.ns, zeta_pp)*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.ale.Yo)
            self.A_pp += InnerProduct(kappa_pp*self.ns, nu_pp)*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.ale.Yo)
            self.A_pp += InnerProduct(1/model.dt*Grad(dX_pp).Trace(), Grad(nu_pp).Trace())*ds(deformation = model.ale.Yo)
            self.A_pp.Assemble()
            self.invA_pp = self.A_pp.mat.Inverse(freedofs = fes_pp.FreeDofs())

            self.F_pp = LinearForm(fes_pp)
            self.F_pp += InnerProduct(self.gfu_norm_vel*model.dt, zeta_pp )*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.ale.Yo)

        elif model.ale.surface_ALE == 'gnz':

            self.A_pp = BilinearForm(fes_pp)
            self.A_pp += InnerProduct(dX_pp*self.ns, zeta_pp)*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.ale.Yo)
            self.A_pp += InnerProduct(kappa_pp*self.ns, nu_pp)*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.ale.Yo)
            self.A_pp += InnerProduct(Grad(dX_pp).Trace(), Grad(nu_pp).Trace())*ds(deformation = model.ale.Yo)
            self.A_pp.Assemble()
            self.invA_pp = self.A_pp.mat.Inverse(freedofs = fes_pp.FreeDofs())

            self.F_pp = LinearForm(fes_pp)
            self.F_pp += InnerProduct(self.gfu_norm_vel*model.dt, zeta_pp )*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.ale.Yo)
            self.F_pp += -1*InnerProduct(self.Ps, Grad(nu_pp).Trace())*ds(deformation = model.ale.Yo)

        elif model.ale.surface_ALE == 'ms':

            def deviatoric(u):
                return Sym(Grad(u).Trace()) - Trace(Sym(Grad(u).Trace()))/model.dim*Id(model.dim)
            ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])

            self.A_pp = BilinearForm(fes_pp, symmetric = True)
            self.A_pp += InnerProduct(dX_pp*self.ns, zeta_pp)*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.ale.Yo)
            self.A_pp += InnerProduct(kappa_pp*self.ns, nu_pp)*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.ale.Yo)
            self.A_pp += InnerProduct(deviatoric(dX_pp), deviatoric(nu_pp))*ds(deformation = model.ale.Yo)
            self.A_pp.Assemble()
            self.invA_pp = self.A_pp.mat.Inverse(freedofs = fes_pp.FreeDofs())

            self.F_pp = LinearForm(fes_pp)
            self.F_pp += InnerProduct(self.gfu_norm_vel*model.dt, zeta_pp )*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.ale.Yo)

        elif model.ale.surface_ALE == 'ms0':

            gfu0 = GridFunction(model.ale.Yo.space)
            X0 = GridFunction(model.ale.Yo.space)
            if model.dim == 2:
                X0.Set(CF((x, y)), definedon = compartment.domain)
            elif model.dim == 3:
                X0.Set(CF((x, y, z)), definedon = compartment.domain)

            def deviatoric(u):
                return Sym(Grad(u).Trace()) - Trace(Sym(Grad(u).Trace()))/model.dim*Id(model.dim)
            ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])

            self.A_pp = BilinearForm(fes_pp, symmetric = True)
            self.A_pp += InnerProduct(dX_pp*self.ns, zeta_pp)*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.ale.Yo)
            self.A_pp += InnerProduct(kappa_pp*self.ns, nu_pp)*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.ale.Yo)
            self.A_pp += InnerProduct(deviatoric(dX_pp), deviatoric(nu_pp))*ds(deformation = gfu0)
            self.A_pp.Assemble()
            self.invA_pp = self.A_pp.mat.Inverse(freedofs = fes_pp.FreeDofs())

            self.F_pp = LinearForm(fes_pp)
            self.F_pp += InnerProduct(self.gfu_norm_vel*model.dt, zeta_pp )*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.ale.Yo)
            self.F_pp += -1*InnerProduct(deviatoric(model.ale.Yo), deviatoric(nu_pp))*ds(deformation = gfu0)
            self.F_pp += -1*InnerProduct(deviatoric(X0), deviatoric(nu_pp))*ds(deformation = gfu0)

        else:

            raise Exception('Surface ALE distribution not known')
        
    def set_normal_velocity(self, coef):
        self.normal_velocity = Field(coef)

    def set_tangential_velocity(self, coef):
        self.tangential_velocity = Field(coef)

    def set_domain_velocity(self, coef):
        self.domain_velocity = Field(coef)
        
    def update(self, model: "CosmosModel", redistribute: bool):

        if self.domain_velocity != None:
            self.ale_displ.Set(self.domain_velocity()*model.time.dt, definedon = self.compartment.domain)
            self.mat_displ.vec.data = self.ale_displ.vec.data
        else:
            if self.normal_velocity == None:
                raise Exception(f'Normal velocity for ALE {self.name} must be set')
            if self.tangential_velocity == None:
                raise Exception(f'Tangential velocity for ALE {self.name} must be set')

            if redistribute:
                self.gfu_norm_vel.Set(self.normal_velocity(), definedon = self.compartment.domain)
                self.ale_displ.vec.data[:] = 0
                self.A_pp.Assemble()
                self.invA_pp.Update()
                self.F_pp.Assemble()

                self.gfu_pp.vec.data = self.invA_pp*self.F_pp.vec
                self.ale_displ.vec.data = self.gfu_pp.components[0].vec.data
            else:
                self.gfu_norm_vel.Set(self.normal_velocity(), definedon = self.compartment.domain)
                self.ale_displ.Set((self.gfu_norm_vel*self.ns + self.Ps*self.tangential_velocity())*model.time.dt, definedon = self.compartment.domain)
            self.mat_displ.Set((self.normal_velocity()*self.ns + self.Ps*self.tangential_velocity())*model.time.dt, definedon = self.compartment.domain)

class CosmosVolALEField:

    def __init__(self, name:str, model:"CosmosModel", compartment:CosmosCompartment):

        self.name = name
        self.model = model
        self.compartment = compartment
        self.domain_velocity = None

        self.fes = VectorH1(self.model.parentmesh, order = self.model.geo_order, 
                        definedon = self.compartment.domain)
        
        self.ale_displ = GridFunction(self.fes)
        self.mat_displ = GridFunction(self.fes)
        self.ale_vel = GridFunction(self.fes)
        self.mat_vel = GridFunction(self.fes)
        
    def set_domain_velocity(self, coef):
        self.domain_velocity = Field(coef)
        
    def update(self, model: "CosmosModel", redistribute: bool):

        if self.domain_velocity == None:
            raise Exception(f'Domain velocity for ALE {self.name} must be set')
        
        self.ale_displ.Set(self.domain_velocity()*model.time.dt, definedon = self.compartment.domain)
        self.mat_displ.vec.data = self.ale_displ.vec.data