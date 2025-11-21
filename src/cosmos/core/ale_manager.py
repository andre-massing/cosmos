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
        self.redistribute = False
        self.ale_elapsed_time = None

        if model.is_bnd:
            self.domain = model.parentmesh.Boundaries('.*')
            self.V = VectorH1(model.parentmesh, order = model.geo_order,
                                        definedon = self.domain)
            gfu = GridFunction(self.V)
            model.parentmesh.SetDeformation(gfu)
        else:
            self.domain = model.parentmesh.Materials('.*')
            self.V = VectorH1(model.parentmesh, order = model.geo_order, definedon = self.domain)
            gfu = GridFunction(self.V)
            model.parentmesh.SetDeformation(gfu)
        
        self.dX = GridFunction(self.V)
        self.prev_dX = []

        if 'redistribute' in  self.params.keys():
            if isinstance(self.params['redistribute'], bool):
                self.redistribute = self.params['redistribute']
            else:
                raise Exception(f'Mesh redistribution for for model {model.name} must be either True or False')
            
        
        if model.is_vol:
            # V0 = VectorH1(model.parentmesh, order = model.geo_order, dirichlet = model.parentmesh.Boundaries('.*'))
            # u, v = V0.TnT()
            # self.A = BilinearForm(V0, symmetric = True)
            # self.A += InnerProduct(grad(u), grad(v))*dx(deformation = self.dX)
            # self.A.Assemble()
            # self.invA = self.A.mat.Inverse(freedofs = V0.FreeDofs())

            V0 = VectorH1(model.parentmesh, order = model.geo_order, dirichlet = model.parentmesh.Boundaries('.*'))
            u, v = V0.TnT()
            self.A = BilinearForm(V0, symmetric = True)
            E, nu = 100, 0.2
            mu  = E / 2 / (1+nu)
            lam = E * nu / ((1+nu)*(1-2*nu))
            def Stress(strain):
                return 2*mu*strain + lam*Trace(strain)*Id(model.dim)    
            self.A += InnerProduct(Stress(Sym(Grad(u))), Sym(Grad(v)))*dx(deformation = self.dX)
            self.A.Assemble()
            self.invA = self.A.mat.Inverse(freedofs = V0.FreeDofs())

            # V0 = VectorH1(model.parentmesh, order = model.geo_order, dirichlet = model.parentmesh.Boundaries('.*'))
            # u, v = V0.TnT()
            # self.A = BilinearForm(V0, symmetric = True)
            # E, nu = 210, 0.2
            # mu  = E / 2 / (1+nu)
            # lam = E * nu / ((1+nu)*(1-2*nu))
            # defgrad = Id(model.dim) + Grad(u)  
            # def psi(F):
            #     E = 0.5*(F.trans*F - Id(model.dim))
            #     return mu*InnerProduct(E,E) + lam/2*Trace(E)**2
            # self.A += Variation(psi(defgrad)*dx(deformation = self.dX))
     
        self.gfu_bnd_ale = GridFunction(self.V)
        self.gfu_bnd_mat = GridFunction(self.V)
        self.gfu_vol_ale = GridFunction(self.V)
        self.gfu_vol_mat = GridFunction(self.V)
        self.gfu_tot_ale = GridFunction(self.V)
        self.gfu_tot_mat = GridFunction(self.V)
        self.wind = GridFunction(self.V)

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

        self.gfu_tot_ale.vec.data[:] = 0
        self.gfu_tot_mat.vec.data[:] = 0
        
        for ale in model.ales:
            ale.update(model, self.redistribute)

        if self.bnd_ales:
            self.gfu_bnd_ale.Set(self.bnd_ale_cf, definedon = model.parentmesh.Boundaries('.*'))
            if model.is_vol:
                self._extend_displacement_to_bulk(self.gfu_bnd_ale)
            self.gfu_tot_ale.vec.data += self.gfu_bnd_ale.vec.data

            self.gfu_bnd_mat.Set(self.bnd_mat_cf, definedon = model.parentmesh.Boundaries('.*'))
            if model.is_vol:
                self._extend_displacement_to_bulk(self.gfu_bnd_mat)
            self.gfu_tot_mat.vec.data += self.gfu_bnd_mat.vec.data

        if self.vol_ales:
            self.gfu_vol_ale.Set(self.vol_ale_cf)
            self.gfu_vol_mat.Set(self.vol_mat_cf)

            self.gfu_tot_ale.vec.data += self.gfu_vol_ale.vec.data
            self.gfu_tot_mat.vec.data += self.gfu_vol_mat.vec.data

        if self.bnd_ales or self.vol_ales:
            self.wind.Set((self.gfu_tot_mat - self.gfu_tot_ale)/model.dt, definedon = self.domain, dual = True)

            self.dX.vec.data += self.gfu_tot_ale.vec.data
            model.parentmesh.deformation.vec.data = self.dX.vec.data

        self.prev_dX.append(self.dX.vec.Copy())
        if len(self.prev_dX)>6:
            self.prev_dX.pop(0)

        stop = time.time()

        self.ale_elapsed_time = stop-start

    def _extend_displacement_to_bulk(self, gfu):

        self.A.Assemble()
        vec = -1*self.A.mat*gfu.vec
        self.invA.Update()
        gfu.vec.data += self.invA*vec

        # Newton(self.A,  gfu, freedofs=gfu.space.FreeDofs(), printing = False, dampfactor=0.1)

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

        self.V = VectorH1(self.model.parentmesh, order = self.model.geo_order, 
                        definedon = self.compartment.domain,
                        dirichlet_bbnd = self.compartment.boundary)
        S = H1(self.model.parentmesh, order = self.model.geo_order, 
                        definedon = self.compartment.domain,
                        dirichlet_bbnd = self.compartment.boundary)
        
        self.ale_displ = GridFunction(self.V)
        self.mat_displ = GridFunction(self.V)
        self.ale_vel = GridFunction(self.V)
        self.mat_vel = GridFunction(self.V)

        self.gfu_norm_vel = GridFunction(S)

        fes_pp = self.V*S
        (dX_pp, kappa_pp), (nu_pp, zeta_pp) = fes_pp.TnT()
        self.gfu_pp = GridFunction(fes_pp)

        ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])

        self.A_pp = BilinearForm(fes_pp, symmetric = True)
        self.A_pp += InnerProduct(dX_pp*self.ns, zeta_pp)*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.dX)
        self.A_pp += InnerProduct(kappa_pp*self.ns, nu_pp)*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.dX)
        self.A_pp += InnerProduct(1/model.dt*Grad(dX_pp).Trace(), Grad(nu_pp).Trace())*ds(deformation = model.dX)
        self.A_pp.Assemble()
        self.invA_pp = self.A_pp.mat.Inverse(freedofs = fes_pp.FreeDofs())

        self.F_pp = LinearForm(fes_pp)
        self.F_pp += InnerProduct(self.gfu_norm_vel*model.dt, zeta_pp )*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.dX)

        # def deviatoric(u):
        #     return Sym(Grad(u).Trace()) - Trace(Sym(Grad(u).Trace()))/model.dim*Id(model.dim)
        # ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        # ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])

        # self.A_pp = BilinearForm(fes_pp, symmetric = True)
        # self.A_pp += InnerProduct(dX_pp*self.ns, zeta_pp)*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.dX)
        # self.A_pp += InnerProduct(kappa_pp*self.ns, nu_pp)*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.dX)
        # self.A_pp += InnerProduct(deviatoric(dX_pp), deviatoric(nu_pp))*ds(deformation = model.dX)
        # self.A_pp.Assemble()
        # self.invA_pp = self.A_pp.mat.Inverse(freedofs = fes_pp.FreeDofs())

        # self.F_pp = LinearForm(fes_pp)
        # self.F_pp += InnerProduct(self.gfu_norm_vel*model.dt, zeta_pp )*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = model.dX)
        
    def set_normal_velocity(self, coef):
        self.normal_velocity = Field(coef)

    def set_tangential_velocity(self, coef):
        self.tangential_velocity = Field(coef)

    def set_domain_velocity(self, coef):
        self.domain_velocity = Field(coef)
        
    def update(self, model: "CosmosModel", redistribute: bool):

        if self.domain_velocity != None:
            self.ale_displ.Set(self.domain_velocity()*model.time.dt, definedon = self.compartment.domain)
            self.mat_displ.Set(self.domain_velocity()*model.time.dt, definedon = self.compartment.domain)
        else:
            if self.normal_velocity == None:
                raise Exception(f'Normal velocity for ALE {self.name} must be set')
            if self.tangential_velocity == None:
                raise Exception(f'Tangential velocity for ALE {self.name} must be set')

            if redistribute:
                self.gfu_norm_vel.Set(self.normal_velocity(), definedon = self.compartment.domain, dual = True)
                self.ale_displ.vec.data[:] = 0
                self.A_pp.Assemble()
                self.invA_pp.Update()
                self.F_pp.Assemble()

                self.gfu_pp.vec.data = self.invA_pp*self.F_pp.vec
                self.ale_displ.vec.data = self.gfu_pp.components[0].vec.data
            else:
                self.gfu_norm_vel.Set(self.normal_velocity(), definedon = self.compartment.domain, dual = True)
                self.ale_displ.Set((self.gfu_norm_vel*self.ns + self.Ps*self.tangential_velocity())*model.time.dt, definedon = self.compartment.domain)
            self.mat_displ.Set((self.normal_velocity()*self.ns + self.Ps*self.tangential_velocity())*model.time.dt, definedon = self.compartment.domain)

            self.ale_vel.Set(self.ale_displ/model.time.dt, definedon = self.compartment.domain)
            self.mat_vel.Set(self.mat_displ/model.time.dt, definedon = self.compartment.domain)


class CosmosVolALEField:

    def __init__(self, name:str, model:"CosmosModel", compartment:CosmosCompartment):

        self.name = name
        self.model = model
        self.compartment = compartment
        self.domain_velocity = None

        self.V = VectorH1(self.model.parentmesh, order = self.model.geo_order, 
                        definedon = self.compartment.domain)
        
        self.ale_displ = GridFunction(self.V)
        self.mat_displ = GridFunction(self.V)
        self.ale_vel = GridFunction(self.V)
        self.mat_vel = GridFunction(self.V)
        
    def set_domain_velocity(self, coef):
        self.domain_velocity = Field(coef)
        
    def update(self, model: "CosmosModel", redistribute: bool):

        if self.domain_velocity == None:
            raise Exception(f'Domain velocity for ALE {self.name} must be set')
        
        self.ale_displ.Set(self.domain_velocity()*model.time.dt, definedon = self.compartment.domain)
        self.mat_displ.Set(self.domain_velocity()*model.time.dt, definedon = self.compartment.domain)
        self.ale_vel.Set(self.ale_displ/model.time.dt, definedon = self.compartment.domain)
        self.mat_vel.Set(self.mat_displ/model.time.dt, definedon = self.compartment.domain)