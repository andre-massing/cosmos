import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.solver import Solver
from cosmos.core.field import InputField, OutputField
from ngsolve.webgui import Draw

class WillmoreBoundaryVPBDF1Model(BasePDEModel):

    def __init__(self, solver:Solver,
                 model_order:int,
                 domain:str = '.*',
                 name:str = 'WillmoreBoundaryVPBDF1Model',
                 input_params = {}):

        super().__init__()

        self.name = name
        self._solver = solver
        self.model_order = model_order

        if (domain in solver.mesh.bnd_markers) or domain == '.*':
            self.domain = solver.ngsmesh.Boundaries(domain)
        else:
            raise Exception('The domain specified for the Model ' + self.name + ' does not exist')

        solver._attach_model(self, self.model_order)

        self.input_params["clamped_bnd"] = ''
        self.input_params["clamped_conormal"] = CF((0,)*self._solver.ngsmesh.dim)
        self.input_params["elasticity_modulus"] = 1
        self.input_params["autoupdate"] = False

        self.set_input_params(input_params)

        if self.input_params["clamped_bnd"]!='':
            raise('Volume preserving algorithm not yet implemented for Willmore flow with clamped boundaries')
        
        if (self.input_params["clamped_bnd"]!='') and \
            (self.input_params["clamped_bnd"] not in self._solver.mesh.bbnd_markers):
            raise ValueError('Clamped boundary conditions are imposed on non-existing BBoundary')

        h = self.cfg.h
        tE = specialcf.tangential(self._solver.ngsmesh.dim)
        self.ns = specialcf.normal(self._solver.ngsmesh.dim)
        self.Ps = Id(self._solver.ngsmesh.dim) - OuterProduct(self.ns, self.ns)
        if self._solver.ngsmesh.dim == 2:
            nE = tE
            tEc = CF((-self.ns[1], self.ns[0]))
        else:
            nE = Cross(self.ns, tE)

        # TODO: Check is this can actually be merged 
        if self.input_params["clamped_bnd"]:
            V1 = VectorH1(self._solver.ngsmesh, order=1, definedon=self.domain,
                    dirichlet_bbnd = self._solver.ngsmesh.BBoundaries(self.input_params["clamped_bnd"]))
        else:
            V1 = VectorH1(self._solver.ngsmesh, order=1,
                        definedon=self.domain)
        V2 = VectorH1(self._solver.ngsmesh, order=1,definedon=self.domain)
        V3 = H1(self._solver.ngsmesh, order=1, definedon=self.domain)
        
        V4 = NumberSpace(self._solver.ngsmesh, definedon=self.domain)
        self.omega_h = GridFunction(V4)

        self.pre_normal = GridFunction(V1)
        self.normal = GridFunction(V1)

        fes = CompressCompound(V1*V2)
        self.A = BilinearForm(fes)
        self.F = LinearForm(fes)

        self.deform = self._solver.mesh.prev_deformation[-1]

        ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
        
        self.ds_lumped = ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = self.deform)
        ds_el_lumped = ds(element_boundary=True, intrules = { SEGM : ir_segm, TRIG: ir_trig })

        fes_mc = Compress(V2)
        kappa_mc, eta_mc = fes_mc.TnT()
        self.A_mc = BilinearForm(fes_mc, symmetric = True)
        self.A_mc += (kappa_mc*eta_mc).Compile(True, True)*self.ds_lumped
        self.A_mc.Assemble()
        self.invA_mc = self.A_mc.mat.Inverse(freedofs = fes_mc.FreeDofs())
        self.F_mc = LinearForm(fes_mc)
        self.F_mc += -InnerProduct(self.Ps, grad(eta_mc).Trace())*ds(deformation = self.deform)
        # if self.input_params["clamped_bnd"]:
        #     if self._solver.ngsmesh.dim == 2:
        #         gfBB = GridFunction(H1(self._solver.ngsmesh, order =1,\
        #                 definedon=self.domain))
        #         gfBB.Set(1, definedon=self._solver.ngsmesh.BBoundaries(self.clamped_bnd))
        #         self.F_mc += InnerProduct(gfBB*nE, eta_mc).Compile(True, True)*ds_el_lumped
        #     elif self._solver.ngsmesh.dim == 3:
        #         gfBB = GridFunction(FacetSurface(self._solver.ngsmesh, order=0, definedon = self.domain))
        #         gfBB.Set(1, definedon=self._solver.ngsmesh.BBoundaries(self.clamped_bnd))
        #         self.F_mc += InnerProduct(nE, eta_mc).Compile(True, True)*gfBB*ds_el_lumped
        if self.input_params["clamped_bnd"]:
            if self._solver.ngsmesh.dim == 2:
                gfBB = GridFunction(H1(self._solver.ngsmesh, order =1,\
                        definedon=self.domain))
                gfBB.Set(1, definedon=self._solver.ngsmesh.BBoundaries(self.input_params["clamped_bnd"]))
                self.F_mc += InnerProduct(gfBB*self.input_params["clamped_conormal"], eta_mc).Compile(True, True)*ds_el_lumped
            elif self._solver.ngsmesh.dim == 3:
                gfBB = GridFunction(FacetSurface(self._solver.ngsmesh, order=0, definedon = self.domain))
                gfBB.Set(1, definedon=self._solver.ngsmesh.BBoundaries(self.input_params["clamped_bnd"]))
                self.F_mc += InnerProduct(self.input_params["clamped_conormal"], eta_mc).Compile(True, True)*gfBB*ds_el_lumped
        self.F_mc.Assemble()

        (trial_D, trial_Y), (test_D, test_Y) = fes.TnT()
        self.gfu = GridFunction(fes)
        self.gfu_old = GridFunction(fes)
        self.gfu_D, self.gfu_Y = self.gfu.components
        self.gfu_D_old, self.gfu_Y_old = self.gfu_old.components
        self.gfu_k = GridFunction(V2)
        self.gfu_k_old = GridFunction(V2)

        self.output_fields["displacement"] = OutputField(self.gfu_D, "displacement", BND)
        self.output_fields["mean_curvature"] = OutputField(self.gfu_k, "mean_curvature", BND)

        self.k0_gfu = GridFunction(V3)
        rhs_gfu = GridFunction(V2)
        self.input_fields["spontaneous_curvature"] = InputField(self.k0_gfu, CF(0), "spontaneous_curvature", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["rhs"] = InputField(rhs_gfu, CF((0,)*self._solver.ngsmesh.dim), "rhs", self._solver.ngsmesh.Boundaries('.*'))
        

        self.X0 = GridFunction(V1)
        if self._solver.ngsmesh.dim == 2:
            self.X0.Set(CF((x,y)), dual = True, definedon=self.domain)
        elif self._solver.ngsmesh.dim == 3:
            self.X0.Set(CF((x, y, z)), dual = True, definedon=self.domain)
        self.displacement_tot = GridFunction(V1)

        self.A += (1/self._solver.time.dt*trial_D*test_D + 1/self.input_params["elasticity_modulus"]*InnerProduct(trial_Y, test_Y)).Compile(True, True)*self.ds_lumped
        self.A += (-InnerProduct(grad(trial_Y).Trace(), grad(test_D).Trace()) + InnerProduct(grad(trial_D).Trace(), grad(test_Y).Trace())).Compile(True, True)*ds(deformation = self.deform)
        
        self.F += (-InnerProduct(self.Ps, grad(test_Y).Trace())).Compile(True, True)*ds(deformation = self.deform)
        self.F += (-1*self.k0_gfu*InnerProduct(self.ns, test_Y)).Compile(True, True)*self.ds_lumped
        self.F += (InnerProduct(rhs_gfu, test_D)).Compile(True, True)*self.ds_lumped

        self.F += (InnerProduct(Trace(grad(self.gfu_Y_old).Trace()),Trace(grad(test_D).Trace()))).Compile(True, True)*ds(deformation = self.deform)
        self.F += (-2*InnerProduct(grad(self.gfu_Y_old).Trace().trans, self.D_s(test_D, self.Ps)*self.Ps.trans)).Compile(True, True)*ds(deformation = self.deform)
        self.F += (-self.input_params["elasticity_modulus"]*InnerProduct(self.k0_gfu*self.gfu_k_old, grad(test_D).Trace().trans*self.ns)).Compile(True, True)*self.ds_lumped
        self.F += (-0.5*InnerProduct(self.input_params["elasticity_modulus"]*(Norm(self.gfu_k_old - self.k0_gfu*self.ns)**2)*self.Ps,grad(test_D).Trace())).Compile(True, True)*self.ds_lumped
        self.F += (InnerProduct(InnerProduct(self.gfu_Y_old, self.gfu_k_old)*self.Ps,grad(test_D).Trace())).Compile(True, True)*self.ds_lumped

        if self.input_params["clamped_bnd"]:
            self.F += (InnerProduct(self.input_params["clamped_conormal"], test_Y)*gfBB).Compile(True, True)*ds_el_lumped
        
        # if self.input_params["clamped_bnd"]:
        #     self.F += (InnerProduct(nE, test_Y)*gfBB).Compile(True, True)*ds_el_lumped

        self.F += (self.omega_h*InnerProduct(self.normal, test_D)).Compile(True, True)*self.ds_lumped

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = fes.FreeDofs())

    def D_s(self, chi, Ps):
        sym = 0.5*Ps*(grad(chi).Trace()+grad(chi).Trace().trans)*Ps
        return sym

    def PreProcess(self):
        
        self.gfu_old.vec.data = self.gfu.vec.data

        if self.input_params['autoupdate'] or self._solver.time.iter == 0:
            self.A_mc.Assemble()
            self.invA_mc.Update()
            self.F_mc.Assemble()
            self.gfu_k_old.vec.data = self.invA_mc*self.F_mc.vec
            self.gfu_Y_old.Set(self.input_params["elasticity_modulus"]*(self.gfu_k_old - self.input_fields["spontaneous_curvature"].gfu*self.ns), dual = True, definedon = self.domain)
            self.gfu_k.vec.data = self.gfu_k_old.vec.data
            self.gfu_Y.vec.data = self.gfu_Y_old.vec.data
            self.pre_normal.Set(self.ns, dual = True, definedon = self.domain)
            self.normal.Set(Normalize(self.pre_normal), dual = True, definedon = self.domain)
        else:
            self.gfu_k_old.vec.data = self.gfu_k.vec.data
            self.gfu_Y_old.vec.data = self.gfu_Y.vec.data

    def Solve(self):

        self.update_input_fields()
        
        self.gfu_D.vec.data[:]=0

        old = 0
        new = 1e5
        iter = 0
        
        while abs(new-old)>1e-8 and iter<10:

            iter += 1
            old = self.omega_h.vec.data[0]

            self.omega_h.vec.data[0] = \
                Integrate(
                    -InnerProduct(grad(self.gfu_Y).Trace(), grad(self.normal).Trace())*ds(deformation = self.deform)
                    -1*InnerProduct(self.input_fields['rhs'].gfu, self.normal)*self.ds_lumped
                    -1*InnerProduct(Trace(grad(self.gfu_Y).Trace()),Trace(grad(self.normal).Trace()))*ds(deformation = self.deform)
                    +2*InnerProduct(grad(self.gfu_Y).Trace().trans, self.D_s(self.normal, self.Ps)*self.Ps.trans)*ds(deformation = self.deform)
                    +self.input_params["elasticity_modulus"]*InnerProduct(self.k0_gfu*self.gfu_k, grad(self.normal).Trace().trans*self.ns)*self.ds_lumped
                    +0.5*InnerProduct(self.input_params["elasticity_modulus"]*(Norm(self.gfu_k - self.k0_gfu*self.ns)**2)*self.Ps,grad(self.normal).Trace())*self.ds_lumped
                    -InnerProduct(InnerProduct(self.gfu_Y, self.gfu_k)*self.Ps,grad(self.normal).Trace())*self.ds_lumped
                , self._solver.ngsmesh)/Integrate(InnerProduct(self.normal, self.normal)*self.ds_lumped, self._solver.ngsmesh)
            
            new = self.omega_h.vec.data[0]

            self.A.Assemble()
            self.invA.Update()
            self.F.Assemble()

            self.gfu.vec.data = self.invA*self.F.vec
            self.gfu_k.Set(1/self.input_params["elasticity_modulus"]*self.gfu_Y + self.input_fields["spontaneous_curvature"].gfu*self.ns, dual = True, definedon = self.domain)

        if iter == 10:
            raise Exception('Convergence not achieved for Volume preserving Willmore flow')

    def PostProcess(self):

        pass

    @property
    def displacement(self):
        return self.gfu_D
    
    @displacement.setter
    def phase(self, cf):
        self.gfu_D.Set(cf, definedon = self.domain)

    @property
    def mean_curvature(self):
        return self.gfu_k
    
    @mean_curvature.setter
    def potential(self, cf):
        self.gfu_k.Set(cf, definedon = self.domain)