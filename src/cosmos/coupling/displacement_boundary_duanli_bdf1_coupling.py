import logging
logger = logging.getLogger(__name__)

import numpy as np
from ngsolve import *
from cosmos.config.parameters import get_config
from cosmos.pde.base import BasePDEModel
from cosmos.core.field import InputField, OutputField
from cosmos.core.solver import Solver
from ngsolve.webgui import Draw

class DisplacementBoundaryDuanLiBDF1Coupling(BasePDEModel):

    def __init__(self, solver:Solver, model_order:int, displacement_field, name:str = 'DisplacementBoundaryBDF1Model',
                 buffer = -1, clamped_bnd = ''):
        
        super().__init__()

        self.name = name
        self._solver = solver
        self.model_order = model_order
        self.buffer = buffer

        solver._attach_model(self, self.model_order)
        self.domain = self._solver.mesh.domain

        V1 = VectorH1(self._solver.ngsmesh, order=self._solver.ngsmesh.GetCurveOrder(), definedon = self.domain,
                      dirichlet_bbnd = self._solver.ngsmesh.BBoundaries(clamped_bnd))
        self.gfu = GridFunction(V1)

        if self.buffer == -1:
            deformation0 = GridFunction(self._solver.mesh.curr_deformation.space)
        elif self.buffer in np.arange(get_config().buffer)+1:
            deformation0 = self._solver.mesh.prev_deformation[-self.buffer]
        else:
            raise Exception('Simulation buffer is too tiny to be compatible with the Coupling buffer')

        V2 = H1(self._solver.ngsmesh, order = self._solver.ngsmesh.GetCurveOrder(),
                definedon = self.domain)

        fes_pp = CompressCompound(V1*V2)
        self.A_pp = BilinearForm(fes_pp, symmetric = True)
        self.F_pp = LinearForm(fes_pp)
        self.gfu_duanli = GridFunction(fes_pp)

        (w, kappa), (eta, mu) = fes_pp.TnT()
        ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
        ns = specialcf.normal(self._solver.ngsmesh.dim)
        Ps = Id(self._solver.ngsmesh.dim) - OuterProduct(ns, ns)
        
        self.A_pp += (InnerProduct(grad(w).Trace(), grad(eta).Trace())).Compile(True, True)*ds(deformation = deformation0)
        self.A_pp += (-1*InnerProduct(kappa*ns, eta)).Compile(True, True)*ds(deformation=self._solver.mesh.curr_deformation)
        self.A_pp += (-1*InnerProduct(w*ns, mu)).Compile(True, True)*ds(deformation=self._solver.mesh.curr_deformation)
        Precond_pp = Preconditioner(self.A_pp, "multigrid")
        self.A_pp.Assemble()
        self.invA_pp = CGSolver(self.A_pp.mat, Precond_pp.mat, maxsteps = 1000)

        self.F_pp += (-1*InnerProduct(Ps, grad(eta).Trace())).Compile(True, True)*ds(deformation = deformation0)
        self.F_pp += (-1*InnerProduct(grad(self._solver.mesh.curr_deformation).Trace(), grad(eta).Trace())).Compile(True, True)*ds(deformation = deformation0)
        self.F_pp += (InnerProduct(grad(deformation0).Trace(), grad(eta).Trace())).Compile(True, True)*ds(deformation = deformation0)

        self.input_fields["displacement"] = InputField(self.gfu, displacement_field, "displacement", self.domain)
        self.output_fields["displacement"] = OutputField(self.gfu, "displacement", BND)

    def PreProcess(self):
        
        pass

    def Solve(self):

        # self._solver.time.advance_tcoef()
        self.update_input_fields()
        self._solver.mesh.curr_deformation.vec.data = self._solver.mesh.prev_deformation[-1].vec.data + self.gfu.vec.data

        self.A_pp.Assemble()
        self.F_pp.Assemble()
        self.gfu_duanli.vec.data = self.invA_pp*self.F_pp.vec
        self._solver.mesh.curr_deformation.vec.data += self.gfu_duanli.components[0].vec.data
        self.gfu.vec.data += self.gfu_duanli.components[0].vec.data

        # self._solver.time.reset_tcoef()

    def PostProcess(self):
        
        self._solver.mesh.update_state()
        for field in self.output_fields.values():
            if field.vtk and self._solver.time.iter % field.sample_rate == 0:
                field.vtk.Do(time = self._solver.time.t.Get(), vb = field.domain)

    @property
    def displacement(self):
        return self.gfu