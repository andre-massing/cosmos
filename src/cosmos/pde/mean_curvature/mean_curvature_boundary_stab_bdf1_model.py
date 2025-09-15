import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.solver import Solver
from cosmos.core.field import InputField, OutputField

class MeanCurvatureBoundaryStabBDF1Model(BasePDEModel):

    def __init__(self, solver:Solver,
                 model_order:int,
                 domain:str = '.*',
                 name:str = 'MeanCurvatureBoundaryStabBDF1Model',
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

        self.input_params["stabilization"] = 0.01

        self.set_input_params(input_params)

        h = self.cfg.h
        tE = specialcf.tangential(self._solver.ngsmesh.dim)
        self.ns = specialcf.normal(self._solver.ngsmesh.dim)
        Ps = Id(self._solver.ngsmesh.dim) - OuterProduct(self.ns, self.ns)
        if self._solver.ngsmesh.dim == 2:
            nE = tE
            tEc = CF((-self.ns[1], self.ns[0]))
        else:
            nE = Cross(self.ns, tE)


        V1 = VectorH1(self._solver.ngsmesh, order=1,
                    definedon=self.domain)
        V2 = VectorH1(self._solver.ngsmesh, order=1,definedon=self.domain)
        V3 = H1(self._solver.ngsmesh, order=1,definedon=self.domain)
        # if self._solver.ngsmesh.dim == 2:
        #     dV = H1(self._solver.ngsmesh, order=1,definedon = self.domain)
        # elif self._solver.ngsmesh.dim == 3:
        dV = NormalFacetSurface(self._solver.ngsmesh, order=0, definedon = self.domain)

        fes = CompressCompound(V1*V2*dV)
        self.A = BilinearForm(fes)
        self.F = LinearForm(fes)

        deform = self._solver.mesh.prev_deformation[-1]

        ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
        
        ds_lumped = ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = deform)
        ds_el_lumped = ds(element_boundary=True, intrules = { SEGM : ir_segm, TRIG: ir_trig })

        (trial_D, trial_k, trial_dk), (test_D, test_k, test_dk) = fes.TnT()
        self.gfu = GridFunction(fes)
        self.gfu_old = GridFunction(fes)
        self.gfu_D, self.gfu_k, _ = self.gfu.components

        self.output_fields["displacement"] = OutputField(self.gfu_D, "displacement", BND)
        self.output_fields["mean_curvature"] = OutputField(self.gfu_k, "mean_curvature", BND)


        rhs_gfu = GridFunction(V2)
        self.input_fields["rhs"] = InputField(rhs_gfu, CF((0,)*self._solver.ngsmesh.dim), "rhs", self._solver.ngsmesh.Boundaries('.*'))

        self.X0 = GridFunction(V1)
        if self._solver.ngsmesh.dim == 2:
            self.X0.Set(CF((x,y)), dual = True, definedon=self.domain)
        elif self._solver.ngsmesh.dim == 3:
            self.X0.Set(CF((x, y, z)), dual = True, definedon=self.domain)
        self.displacement_tot = GridFunction(V1)

        self.A += (1/self._solver.time.dt*trial_D*test_D + InnerProduct(trial_k, test_k))*ds_lumped
        self.A += (InnerProduct(grad(trial_D).Trace(), grad(test_k).Trace()))*ds(deformation = deform)
        self.A += (-InnerProduct(trial_k, test_D))*ds_lumped
        jump_dkappadn = (trial_k.Trace().Deriv()*nE-trial_dk.Trace())
        jump_detadn = (test_k.Trace().Deriv()*nE-test_dk.Trace())
        self.A += (self.input_params["stabilization"]*h*InnerProduct(jump_dkappadn,jump_detadn))*ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, 
                                                                                                    element_boundary=True, deformation = deform)
        
        self.F += (-InnerProduct(Ps, grad(test_k).Trace()))*ds(deformation = deform)
        self.F += (InnerProduct(rhs_gfu, test_D))*ds_lumped

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = fes.FreeDofs())

    def PreProcess(self):
        
        self.gfu_old.vec.data = self.gfu.vec.data

    def Solve(self):

        self.update_input_fields()

        self.A.Assemble()
        self.invA.Update()
        self.F.Assemble()

        self.gfu.vec.data = self.invA*self.F.vec

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