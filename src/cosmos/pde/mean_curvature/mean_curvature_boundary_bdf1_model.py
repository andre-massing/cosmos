import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.solver import Solver
from cosmos.core.field import InputField, OutputField

class MeanCurvatureBoundaryBDF1Model(BasePDEModel):

    def __init__(self, solver:Solver,
                 model_order:int,
                 domain:str = '.*',
                 name:str = 'MeaCurvatureBoundaryBDF1Model',
                 input_params = {}):

        super().__init__()

        self.name = name
        self._solver = solver
        self.model_order = model_order

        if (domain in solver.mesh.bnd_markers) or domain == '.*':
            self.domain = solver.ngsmesh.Boundaries(domain)
        else:
            logger.error('The domain specified for the Model ', self.name, ' does not exist')

        solver._attach_model(self, self.model_order)

        self.input_params["kappa"] = 1
        self.set_input_params(input_params)

        self.ns = specialcf.normal(self._solver.ngsmesh.dim)
        Ps = Id(self._solver.ngsmesh.dim) - OuterProduct(self.ns, self.ns)

        V1 = VectorH1(self._solver.ngsmesh, order=1,
                    definedon=self.domain)

        fes = CompressCompound(V1*V1)
        self.A = BilinearForm(fes)
        self.F = LinearForm(fes)

        deform = self._solver.mesh.prev_deformation[-1]

        ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
        
        ds_lumped = ds(intrules = { SEGM : ir_segm, TRIG: ir_trig }, deformation = deform)

        (trial_D, trial_k), (test_D, test_k) = fes.TnT()
        self.gfu = GridFunction(fes)
        self.gfu_old = GridFunction(fes)
        self.gfu_D, self.gfu_k = self.gfu.components

        self.output_fields["displacement"] = OutputField(self.gfu_D, "displacement", BND)
        self.output_fields["mean_curvature"] = OutputField(self.gfu_k, "mean_curvature", BND)

        rhs_gfu = GridFunction(V1)
        self.input_fields["rhs"] = InputField(rhs_gfu, CF((0,)*self._solver.ngsmesh.dim), "rhs", self._solver.ngsmesh.Boundaries('.*'))

        self.X0 = GridFunction(V1)
        if self._solver.ngsmesh.dim == 2:
            self.X0.Set(CF((x,y)), dual = True, definedon=self.domain)
        elif self._solver.ngsmesh.dim == 3:
            self.X0.Set(CF((x, y, z)), dual = True, definedon=self.domain)
        self.displacement_tot = GridFunction(V1)

        self.A += (1/self._solver.time.dt*trial_D*test_D + InnerProduct(trial_k, test_k))*ds_lumped
        self.A += (InnerProduct(grad(trial_D).Trace(), grad(test_k).Trace()))*ds(deformation = deform)
        self.A += (-self.input_params["kappa"]*InnerProduct(trial_k, test_D))*ds_lumped
        
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