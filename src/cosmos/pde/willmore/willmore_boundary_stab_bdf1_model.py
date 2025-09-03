import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.solver import Solver
from cosmos.core.field import InputField, OutputField

class WillmoreBoundaryStabBDF1Model(BasePDEModel):

    def __init__(self, solver:Solver,
                 model_order:int,
                 domain:str = '.*',
                 name:str = 'WillmoreBoundaryStabBDF1Model',
                 input_params = {}):

        super().__init__()

        self.name = name
        self._solver = solver
        self.model_order = model_order

        if (domain in solver.mesh.bnd_markers) or domain == '.*':
            self.domain = solver.ngsmesh.Boundaries(domain)
        else:
            logger.error('The domain specified for the Model ', self.name, ' does not exist')

        if self._solver.ngsmesh.dim == 2:
            raise Exception('Stabilization not implemented for 1D manifolds!')

        solver._attach_model(self, self.model_order)

        self.input_params["clamped_bnd"] = ''
        self.input_params["clamped_conormal"] = CF((0,)*self._solver.ngsmesh.dim)
        self.input_params["elasticity_modulus"] = 1
        self.input_params["stabilization"] = 0.01
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
        V3 = H1(self._solver.ngsmesh, order=1,definedon=self.domain)
        # if self._solver.ngsmesh.dim == 2:
        #     dV = H1(self._solver.ngsmesh, order=1,definedon = self.domain)
        # elif self._solver.ngsmesh.dim == 3:
        dV = NormalFacetSurface(self._solver.ngsmesh, order=0, definedon = self.domain)

        fes = CompressCompound(V1*V2)
        self.A = BilinearForm(fes)
        self.F = LinearForm(fes)

        deform = self._solver.mesh.prev_deformation[-1]

        ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
        
        ds_lumped = ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = deform)
        ds_el_lumped = ds(element_boundary=True, intrules = { SEGM : ir_segm, TRIG: ir_trig })

        fes_mc = CompressCompound(V2*dV)
        self.gfu_mc = GridFunction(fes_mc)
        (kappa_mc, dkappa_mc), (eta_mc, deta_mc) = fes_mc.TnT()
        jump_dkappadn = (kappa_mc.Trace().Deriv()*nE-dkappa_mc.Trace())
        jump_detadn = (eta_mc.Trace().Deriv()*nE-deta_mc.Trace())
        self.A_mc = BilinearForm(fes_mc, symmetric = True)
        self.A_mc += (kappa_mc*eta_mc).Compile(True, True)*ds_lumped
        self.A_mc += (self.input_params["stabilization"]*h*InnerProduct(jump_dkappadn,jump_detadn))*ds(element_boundary=True)
        self.F_mc = LinearForm(fes_mc)
        self.F_mc += -InnerProduct(Ps, grad(eta_mc).Trace())*ds
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
            gfBB = GridFunction(FacetSurface(self._solver.ngsmesh, order=0, definedon = self.domain))
            gfBB.Set(1, definedon=self._solver.ngsmesh.BBoundaries(self.input_params["clamped_bnd"]))
            self.F_mc += InnerProduct(self.input_params["clamped_conormal"], eta_mc).Compile(True, True)*gfBB*ds_el_lumped
            self.A_mc += -1*gfBB*(self.input_params["stabilization"]*h*InnerProduct(jump_dkappadn,jump_detadn))*ds(element_boundary=True)
            self.A_mc +=  (gfBB*dkappa_mc.Trace()*deta_mc.Trace())*ds(element_boundary=True)
        self.A_mc.Assemble()
        self.invA_mc = self.A_mc.mat.Inverse(freedofs = fes_mc.FreeDofs())
        self.F_mc.Assemble()

        (trial_D, trial_Y), (test_D, test_Y) = fes.TnT()
        self.gfu = GridFunction(fes)
        self.gfu_old = GridFunction(fes)
        self.gfu_D, self.gfu_Y = self.gfu.components
        self.gfu_k = GridFunction(V2)

        self.output_fields["displacement"] = OutputField(self.gfu_D, "displacement", BND)
        self.output_fields["mean_curvature"] = OutputField(self.gfu_k, "mean_curvature", BND)

        k0_gfu = GridFunction(V3)
        rhs_gfu = GridFunction(V2)
        self.input_fields["spontaneous_curvature"] = InputField(k0_gfu, CF((0)), "spontaneous_curvature", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["rhs"] = InputField(rhs_gfu, CF((0,)*self._solver.ngsmesh.dim), "rhs", self._solver.ngsmesh.Boundaries('.*'))
        

        self.X0 = GridFunction(V1)
        if self._solver.ngsmesh.dim == 2:
            self.X0.Set(CF((x,y)), dual = True, definedon=self.domain)
        elif self._solver.ngsmesh.dim == 3:
            self.X0.Set(CF((x, y, z)), dual = True, definedon=self.domain)
        self.displacement_tot = GridFunction(V1)


        self.A += (1/self._solver.time.dt*trial_D*test_D + 1/self.input_params["elasticity_modulus"]*InnerProduct(trial_Y, test_Y)).Compile(True, True)*ds_lumped
        self.A += (-InnerProduct(grad(trial_Y).Trace(), grad(test_D).Trace()) + InnerProduct(grad(trial_D).Trace(), grad(test_Y).Trace())).Compile(True, True)*ds(deformation = deform)

        
        self.F += (-InnerProduct(Ps, grad(test_Y).Trace())).Compile(True, True)*ds(deformation = deform)
        self.F += (-1*k0_gfu*InnerProduct(self.ns, test_Y)).Compile(True, True)*ds_lumped
        self.F += (InnerProduct(rhs_gfu, test_D)).Compile(True, True)*ds_lumped

        def D_s(chi, Ps):
            sym = 0.5*Ps*(grad(chi).Trace()+grad(chi).Trace().trans)*Ps
            return sym
        
        self.F += (InnerProduct(Trace(grad(self.gfu_Y).Trace()),Trace(grad(test_D).Trace()))).Compile(True, True)*ds(deformation = deform)
        self.F += (-2*InnerProduct(grad(self.gfu_Y).Trace().trans, D_s(test_D, Ps)*Ps.trans)).Compile(True, True)*ds(deformation = deform)
        self.F += (-self.input_params["elasticity_modulus"]*InnerProduct(k0_gfu*self.gfu_k, grad(test_D).Trace().trans*self.ns)).Compile(True, True)*ds_lumped
        self.F += (-0.5*InnerProduct(self.input_params["elasticity_modulus"]*(Norm(self.gfu_k - k0_gfu*self.ns)**2)*Ps,grad(test_D).Trace())).Compile(True, True)*ds_lumped
        self.F += (InnerProduct(InnerProduct(self.gfu_Y, self.gfu_k)*Ps,grad(test_D).Trace())).Compile(True, True)*ds_lumped

        if self.input_params["clamped_bnd"]:
            self.F += (InnerProduct(self.input_params["clamped_conormal"], test_Y)*gfBB).Compile(True, True)*ds_el_lumped
        
        # if self.input_params["clamped_bnd"]:
        #     self.F += (InnerProduct(nE, test_Y)*gfBB).Compile(True, True)*ds_el_lumped

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = fes.FreeDofs())

    def PreProcess(self):
        
        self.gfu_old.vec.data = self.gfu.vec.data

        if self.input_params['autoupdate'] or self._solver.time.iter == 0:
            self.A_mc.Assemble()
            self.invA_mc.Update()
            self.F_mc.Assemble()
            self.gfu_mc.vec.data = self.invA_mc*self.F_mc.vec
            self.gfu_k.vec.data = self.gfu_mc.components[0].vec.data
            self.gfu_Y.Set(self.input_params["elasticity_modulus"]*(self.gfu_k - self.input_fields["spontaneous_curvature"].gfu*self.ns), dual = True, definedon = self.domain)

    def Solve(self):

        self.update_input_fields()

        self.A.Assemble()
        self.invA.Update()
        self.F.Assemble()

        self.gfu.vec.data = self.invA*self.F.vec

        self.gfu_k.Set(1/self.input_params["elasticity_modulus"]*self.gfu_Y + self.input_fields["spontaneous_curvature"].gfu*self.ns, dual = True, definedon = self.domain)

    def PostProcess(self):
        
        for field in self.output_fields.values():
            if field.vtk and self._solver.time.iter % field.sample_rate == 0:
                field.vtk.Do(time = self._solver.time.t.Get(), vb = field.domain)

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