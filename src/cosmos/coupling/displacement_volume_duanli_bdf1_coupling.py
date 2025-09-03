import logging
logger = logging.getLogger(__name__)

import numpy as np
from ngsolve import *
from cosmos.config.parameters import get_config
from cosmos.pde.base import BasePDEModel
from cosmos.core.field import InputField, OutputField
from cosmos.core.solver import Solver
from ngsolve.webgui import Draw

class DisplacementVolumeDuanLiBDF1Coupling(BasePDEModel):

    def __init__(self, solver:Solver, model_order:int, displacement_field, 
                 name:str ='DisplacementVolumeBDF1Model', deformation_type = 'hyperelastic',
                 buffer = -1, clamped_bnd = '', redistribute = False):
        
        super().__init__()

        self.name = name
        self._solver = solver
        self.model_order = model_order
        self.buffer = buffer
        self.deformation_type = deformation_type
        self.redistribute = redistribute

        solver._attach_model(self, self.model_order)
        self.domain = self._solver.mesh.domain        
        self.gfu = GridFunction(self._solver.mesh.curr_deformation.space)

        if self.buffer == -1:
            deformation0 = GridFunction(self._solver.mesh.curr_deformation.space)
        elif self.buffer in np.arange(get_config().buffer)+1:
            deformation0 = self._solver.mesh.prev_deformation[-self.buffer]
        else:
            raise Exception('Simulation buffer is too tiny to be compatible with the Coupling buffer')
        
        V1 = Compress(VectorH1(self._solver.ngsmesh, order=self._solver.ngsmesh.GetCurveOrder(),
                      definedon = self._solver.ngsmesh.Boundaries('.*'), 
                      dirichlet_bbnd = self._solver.ngsmesh.BBoundaries(clamped_bnd)))
        gfu_bnd = GridFunction(V1)
        self.gfu_bnd_duanli = GridFunction(V1)
        V2 = Compress(H1(self._solver.ngsmesh, order = self._solver.ngsmesh.GetCurveOrder(),
                definedon = self._solver.ngsmesh.Boundaries('.*')))

        fes_pp = V1*V2
        self.A_pp = BilinearForm(fes_pp, symmetric = True)
        self.F_pp = LinearForm(fes_pp)
        self.gfu_duanli = GridFunction(fes_pp)

        (w, kappa), (eta, mu) = fes_pp.TnT()
        ns = specialcf.normal(self._solver.ngsmesh.dim)
        Ps = Id(self._solver.ngsmesh.dim) - OuterProduct(ns, ns)
        
        self.A_pp += (InnerProduct(grad(w).Trace(), grad(eta).Trace())).Compile(True, True)*ds(deformation = deformation0)
        self.A_pp += (-1*InnerProduct(kappa*ns, eta)).Compile(True, True)*ds(deformation=self.gfu_bnd_duanli)
        self.A_pp += (-1*InnerProduct(w*ns, mu)).Compile(True, True)*ds(deformation=self.gfu_bnd_duanli)
        self.A_pp.Assemble()
        self.invA_pp = self.A_pp.mat.Inverse(freedofs = fes_pp.FreeDofs())

        self.F_pp += (-1*InnerProduct(Ps, grad(eta).Trace())).Compile(True, True)*ds(deformation = deformation0)
        self.F_pp += (-1*InnerProduct(grad(self.gfu_bnd_duanli).Trace(), grad(eta).Trace())).Compile(True, True)*ds(deformation = deformation0)
        self.F_pp += (InnerProduct(grad(deformation0).Trace(), grad(eta).Trace())).Compile(True, True)*ds(deformation = deformation0)

        self.input_fields["displacement"] = InputField(gfu_bnd, CF((0,)*self._solver.ngsmesh.dim), "displacement", self._solver.ngsmesh.Boundaries('.*'))
        self.set_input_fields({"displacement": displacement_field})
        self.output_fields["displacement"] = OutputField(self.gfu, "displacement", VOL)

        vol_space = VectorH1(self._solver.ngsmesh, order=self._solver.ngsmesh.GetCurveOrder(),
                             definedon = self._solver.ngsmesh.Materials('.*'), dirichlet = self._solver.ngsmesh.Boundaries('.*'))
        u, v = vol_space.TnT()
        self.A = BilinearForm(vol_space, symmetric = True)

        self.A += InnerProduct(grad(u), grad(v))*dx
        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = vol_space.FreeDofs())

    def PreProcess(self):
        
        pass

    def Solve(self):

        # self._solver.time.advance_tcoef()
        self.update_input_fields()
        self.gfu_bnd_duanli.Set(self.input_fields["displacement"].gfu + self._solver.mesh.prev_deformation[-1], dual=True, definedon=self._solver.ngsmesh.Boundaries('.*'))

        if self.redistribute:
            self.A_pp.Assemble()
            self.invA_pp.Update()
            self.F_pp.Assemble()
            self.gfu_duanli.vec.data = self.invA_pp*self.F_pp.vec
            self.gfu_bnd_duanli.vec.data += self.gfu_duanli.components[0].vec.data

        self.gfu.Set(self.gfu_bnd_duanli-self._solver.mesh.prev_deformation[-1], dual = True, definedon = self._solver.ngsmesh.Boundaries('.*'))

        self.A.Assemble()
        self.invA.Update()
        res = -1*self.A.mat*self.gfu.vec
        self.gfu.vec.data += self.invA*res
        self._solver.mesh.curr_deformation.vec.data = self._solver.mesh.prev_deformation[-1].vec.data + self.gfu.vec.data
        # self._solver.time.reset_tcoef()

    def PostProcess(self):
        
        self._solver.mesh.update_state()

    @property
    def displacement(self):
        return self.gfu