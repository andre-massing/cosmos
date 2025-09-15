import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.solver import Solver
from cosmos.core.field import InputField, OutputField
from ngsolve.webgui import Draw

class WillmoreBoundaryInexBDF1Model(BasePDEModel):

    def __init__(self, solver:Solver,
                 model_order:int,
                 domain:str = '.*',
                 name:str = 'WillmoreBoundaryInexBDF1Model',
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
        self.input_params["autoupdate"] = False

        self.set_input_params(input_params)
        if (self.input_params["clamped_bnd"]!='') and \
            (self.input_params["clamped_bnd"] not in self._solver.mesh.bbnd_markers):
            raise ValueError('Clamped boundary conditions are imposed on non-existing BBoundary')

        h = self.cfg.h
        tE = specialcf.tangential(self._solver.ngsmesh.dim)
        self.ns = specialcf.normal(self._solver.ngsmesh.dim)
        Ps = Id(self._solver.ngsmesh.dim) - OuterProduct(self.ns, self.ns)
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

        fes = CompressCompound(V1*V2*V3)
        self.A = BilinearForm(fes)
        self.F = LinearForm(fes)

        deform = self._solver.mesh.prev_deformation[-1]

        ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
        
        ds_lumped = ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = deform)
        ds_el_lumped = ds(element_boundary=True, intrules = { SEGM : ir_segm, TRIG: ir_trig })

        fes_mc = Compress(V2)
        kappa_mc, eta_mc = fes_mc.TnT()
        self.A_mc = BilinearForm(fes_mc, symmetric = True)
        self.A_mc += (kappa_mc*eta_mc)*ds_lumped
        self.A_mc.Assemble()
        self.invA_mc = self.A_mc.mat.Inverse(freedofs = fes_mc.FreeDofs())
        self.F_mc = LinearForm(fes_mc)
        self.F_mc += -InnerProduct(Ps, grad(eta_mc).Trace())*ds(deformation = deform)
        # if self.input_params["clamped_bnd"]:
        #     if self._solver.ngsmesh.dim == 2:
        #         gfBB = GridFunction(H1(self._solver.ngsmesh, order =1,\
        #                 definedon=self.domain))
        #         gfBB.Set(1, definedon=self._solver.ngsmesh.BBoundaries(self.clamped_bnd))
        #         self.F_mc += InnerProduct(gfBB*nE, eta_mc)*ds_el_lumped
        #     elif self._solver.ngsmesh.dim == 3:
        #         gfBB = GridFunction(FacetSurface(self._solver.ngsmesh, order=0, definedon = self.domain))
        #         gfBB.Set(1, definedon=self._solver.ngsmesh.BBoundaries(self.clamped_bnd))
        #         self.F_mc += InnerProduct(nE, eta_mc)*gfBB*ds_el_lumped
        if self.input_params["clamped_bnd"]:
            if self._solver.ngsmesh.dim == 2:
                gfBB = GridFunction(H1(self._solver.ngsmesh, order =1,\
                        definedon=self.domain))
                gfBB.Set(1, definedon=self._solver.ngsmesh.BBoundaries(self.input_params["clamped_bnd"]))
                self.F_mc += InnerProduct(gfBB*self.input_params["clamped_conormal"], eta_mc)*ds_el_lumped
            elif self._solver.ngsmesh.dim == 3:
                gfBB = GridFunction(FacetSurface(self._solver.ngsmesh, order=0, definedon = self.domain))
                gfBB.Set(1, definedon=self._solver.ngsmesh.BBoundaries(self.input_params["clamped_bnd"]))
                self.F_mc += InnerProduct(self.input_params["clamped_conormal"], eta_mc)*gfBB*ds_el_lumped
        self.F_mc.Assemble()

        (trial_D, trial_Y, trial_lam), (test_D, test_Y, test_lam) = fes.TnT()
        self.gfu = GridFunction(fes)
        self.gfu_old = GridFunction(fes)
        self.gfu_D, self.gfu_Y, self.gfu_lam = self.gfu.components
        self.gfu_D_old, self.gfu_Y_old, self.gfu_lam_old = self.gfu_old.components
        self.gfu_k = GridFunction(V2)
        self.gfu_k_old = GridFunction(V2)

        self.A_mc.Assemble()
        self.invA_mc.Update()
        self.F_mc.Assemble()
        self.gfu_k_old.vec.data = self.invA_mc*self.F_mc.vec
        self.gfu_k.vec.data = self.gfu_k_old.vec.data

        self.output_fields["displacement"] = OutputField(self.gfu_D, "displacement", BND)
        self.output_fields["mean_curvature"] = OutputField(self.gfu_k, "mean_curvature", BND)
        self.output_fields["multiplier"] = OutputField(self.gfu_lam, "mean_curvature", BND)

        k0_gfu = GridFunction(V3)
        rhs_gfu = GridFunction(V2)
        kappa_gfu = GridFunction(V3)
        self.input_fields["spontaneous_curvature"] = InputField(k0_gfu, CF(0), "spontaneous_curvature", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["rhs"] = InputField(rhs_gfu, CF((0,)*self._solver.ngsmesh.dim), "rhs", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["elasticity_modulus"] = InputField(kappa_gfu, CF(1), "elasticity_modulus", self._solver.ngsmesh.Boundaries('.*'))
        kappa_gfu.Set(1, definedon = self.domain)
        

        self.X0 = GridFunction(V1)
        if self._solver.ngsmesh.dim == 2:
            self.X0.Set(CF((x,y)), dual = True, definedon=self.domain)
        elif self._solver.ngsmesh.dim == 3:
            self.X0.Set(CF((x, y, z)), dual = True, definedon=self.domain)
        self.displacement_tot = GridFunction(V1)

        self.A += (1/self._solver.time.dt*trial_D*test_D + 1/kappa_gfu*InnerProduct(trial_Y, test_Y))*ds_lumped
        self.A += (-InnerProduct(grad(trial_Y).Trace(), grad(test_D).Trace()) + InnerProduct(grad(trial_D).Trace(), grad(test_Y).Trace()))*ds(deformation = deform)
        
        self.F += (-InnerProduct(Ps, grad(test_Y).Trace()))*ds(deformation = deform)
        self.F += (-1*k0_gfu*InnerProduct(self.ns, test_Y))*ds_lumped
        self.F += (InnerProduct(rhs_gfu, test_D))*ds_lumped

        def D_s(chi, Ps):
            sym = 0.5*Ps*(grad(chi).Trace()+grad(chi).Trace().trans)*Ps
            return sym

        self.F += (InnerProduct(Trace(grad(self.gfu_Y_old).Trace()),Trace(grad(test_D).Trace())))*ds(deformation = deform)
        self.F += (-2*InnerProduct(grad(self.gfu_Y_old).Trace().trans, D_s(test_D, Ps)*Ps.trans))*ds(deformation = deform)
        self.F += (-kappa_gfu*InnerProduct(k0_gfu*self.gfu_k_old, grad(test_D).Trace().trans*self.ns))*ds_lumped
        self.F += (-0.5*InnerProduct(kappa_gfu*(Norm(self.gfu_k_old - k0_gfu*self.ns)**2)*Ps,grad(test_D).Trace()))*ds_lumped
        self.F += (InnerProduct(InnerProduct(self.gfu_Y_old, self.gfu_k_old)*Ps,grad(test_D).Trace()))*ds_lumped

        if self.input_params["clamped_bnd"]:
            self.F += (InnerProduct(self.input_params["clamped_conormal"], test_Y)*gfBB)*ds_el_lumped
        
        # if self.input_params["clamped_bnd"]:
        #     self.F += (InnerProduct(nE, test_Y)*gfBB)*ds_el_lumped

        # self.A += (-1*trial_lam*self.gfu_k*test_D)*ds_lumped
        # self.A += (InnerProduct(grad(trial_lam).Trace(), grad(test_lam).Trace()))*ds(deformation = deform)
        # self.A += (trial_lam*Norm(self.gfu_k)**2*test_lam)*ds_lumped
        # self.F += -InnerProduct(grad(self.gfu_Y).Trace(), grad(self.gfu_k).Trace())*test_lam*ds(deformation = deform) \
        #             -1*InnerProduct(self.input_fields['rhs'].gfu, self.gfu_k)*test_lam*ds_lumped \
        #             -1*InnerProduct(Trace(grad(self.gfu_Y).Trace()),Trace(grad(self.gfu_k).Trace()))*test_lam*ds(deformation = deform) \
        #             +2*InnerProduct(grad(self.gfu_Y).Trace().trans, D_s(self.gfu_k, Ps)*Ps.trans)*test_lam*ds(deformation = deform) \
        #             +self.input_fields["elasticity_modulus"].gfu*InnerProduct(k0_gfu*self.gfu_k, grad(self.gfu_k).Trace().trans*self.ns)*test_lam*ds_lumped \
        #             +0.5*InnerProduct(self.input_fields["elasticity_modulus"].gfu*(Norm(self.gfu_k - k0_gfu*self.ns)**2)*Ps,grad(self.gfu_k).Trace())*test_lam*ds_lumped \
        #             -InnerProduct(InnerProduct(self.gfu_Y, self.gfu_k)*Ps,grad(self.gfu_k).Trace())*test_lam*ds_lumped \
        
        self.A += (-1*trial_lam*self.gfu_k_old*test_D)*ds_lumped
        self.A += (InnerProduct(grad(trial_lam).Trace(), grad(test_lam).Trace()))*ds(deformation = deform)
        self.A += (trial_lam*Norm(self.gfu_k_old)**2*test_lam)*ds_lumped
        self.F += -InnerProduct(grad(self.gfu_Y_old).Trace(), grad(self.gfu_k_old).Trace())*test_lam*ds(deformation = deform) \
                    -1*InnerProduct(self.input_fields['rhs'].gfu, self.gfu_k_old)*test_lam*ds_lumped \
                    -1*InnerProduct(Trace(grad(self.gfu_Y_old).Trace()),Trace(grad(self.gfu_k_old).Trace()))*test_lam*ds(deformation = deform) \
                    +2*InnerProduct(grad(self.gfu_Y_old).Trace().trans, D_s(self.gfu_k_old, Ps)*Ps.trans)*test_lam*ds(deformation = deform) \
                    +self.input_fields["elasticity_modulus"].gfu*InnerProduct(k0_gfu*self.gfu_k_old, grad(self.gfu_k_old).Trace().trans*self.ns)*test_lam*ds_lumped \
                    +0.5*InnerProduct(self.input_fields["elasticity_modulus"].gfu*(Norm(self.gfu_k_old - k0_gfu*self.ns)**2)*Ps,grad(self.gfu_k_old).Trace())*test_lam*ds_lumped \
                    -InnerProduct(InnerProduct(self.gfu_Y_old, self.gfu_k_old)*Ps,grad(self.gfu_k_old).Trace())*test_lam*ds_lumped \
        

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = fes.FreeDofs())

    def PreProcess(self):
        
        self.gfu_old.vec.data = self.gfu.vec.data
        self.gfu_k_old.vec.data = self.gfu_k.vec.data

        if self.input_params['autoupdate'] or self._solver.time.iter == 0:
            self.A_mc.Assemble()
            self.invA_mc.Update()
            self.F_mc.Assemble()
            self.gfu_k_old.vec.data = self.invA_mc*self.F_mc.vec
            self.gfu_Y_old.Set(self.input_fields["elasticity_modulus"].gfu*(self.gfu_k_old - self.input_fields["spontaneous_curvature"].gfu*self.ns), dual = True, definedon = self.domain)
            self.gfu_k.vec.data = self.gfu_k_old.vec.data
            self.gfu_Y.vec.data = self.gfu_Y_old.vec.data
        else:
            self.gfu_k_old.vec.data = self.gfu_k.vec.data
            self.gfu_old.vec.data = self.gfu.vec.data

    def Solve(self):

        self.update_input_fields()

        # old = self.gfu_lam.vec.Copy()
        # old.data[:] = 1e5
        # iter = 0
        # maxiter = 10

        # while Norm(old-self.gfu_lam.vec)>1e-8 and iter<maxiter:

        #     iter += 1
        #     old = self.gfu_lam.vec.Copy()

        #     self.A.Assemble()
        #     self.invA.Update()
        #     self.F.Assemble()

        #     self.gfu.vec.data = self.invA*self.F.vec
        #     self.gfu_k.Set(1/self.input_fields["elasticity_modulus"].gfu*self.gfu_Y + self.input_fields["spontaneous_curvature"].gfu*self.ns, dual = True, definedon = self.domain)
        
        # print(iter)
        # if iter == maxiter:
        #     raise Exception('Convergence not achieved for inextensible area Willmore flow')

        self.A.Assemble()
        self.invA.Update()
        self.F.Assemble()

        self.gfu.vec.data = self.invA*self.F.vec
        self.gfu_k.Set(1/self.input_fields["elasticity_modulus"].gfu*self.gfu_Y + self.input_fields["spontaneous_curvature"].gfu*self.ns, dual = True, definedon = self.domain)

    def PostProcess(self):

        pass

    @property
    def displacement(self):
        return self.gfu_D

    @property
    def mean_curvature(self):
        return self.gfu_k

    @property
    def multiplier(self):
        return self.gfu_lam