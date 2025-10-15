import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.solver import Solver
from cosmos.core.field import InputField, OutputField
from cosmos.core.utils import MandBP
from ngsolve.webgui import Draw

class CahnHilliardVolumeAlandBDF1Model(BasePDEModel):

    def __init__(self, solver:Solver,
                 model_order:int,
                 domain:str = '.*',
                 name:str = 'CahnHilliardVolumeAlandBDF1Model',
                 input_params = {}):
        
        super().__init__()
        
        self.VorB = VOL
        self.name = name
        self._solver = solver
        self.model_order = model_order

        if solver.mesh.is_bnd:
            logger.error('The mesh is only a surface, use ACahnHilliardBoundaryBDF1Model')

        if (domain in solver.mesh.vol_markers) or domain == '.*':
            self.domain = solver.ngsmesh.Materials(domain)
        else:
            raise Exception('The domain specified for the Model '+ self.name+ ' does not exist')

        solver._attach_model(self, self.model_order)

        ns = specialcf.normal(self._solver.ngsmesh.dim)
        self.input_params["periodic"] = False
        self.input_params["mass_preserving"] = False
        self.input_params["bounds"] = None
        self.input_params["fes_order"] = 1
        self.input_params["u0"] = CF(0)
        self.input_params["w0"] = CF(0)
        self.input_params["M"] = 1
        self.input_params["epsilon"] = 1
        self.input_params["sigma"] = 1
        self.input_params["Neu_bnd_phase"] = ''
        self.input_params["Neu_bnd_potential"] = ''

        self.set_input_params(input_params)
        
        _fes = H1(self._solver.ngsmesh, order = self.input_params["fes_order"], 
                                            definedon = self.domain)
        if self.input_params["periodic"]:
            fes = Compress(Periodic(_fes))*Compress(Periodic(_fes))
            fes_vector = Compress(Periodic(VectorH1(self._solver.ngsmesh, order = self.input_params["fes_order"], 
                                            definedon = self.domain)))
        else:
            fes = _fes*_fes
            fes_vector = Compress(VectorH1(self._solver.ngsmesh, order = self.input_params["fes_order"], 
                                   definedon = self.domain))
          
        (trial_u, trial_w), (test_u, test_w) = fes.TnT()
        self.A = BilinearForm(fes)
        self.F = LinearForm(fes)
        
        self.gfu = GridFunction(fes)
        self.gfu_u, self.gfu_w = self.gfu.components
        self.gfu_old = GridFunction(fes)
        self.gfu_u_old, self.gfu_w_old = self.gfu_old.components
        self.gfu_u.Set(self.input_params['u0'], dual = True)
        self.output_fields["phase"] = OutputField(self.gfu_u, "phase", VOL)
        self.gfu_w.Set(self.input_params['w0'], dual = True)
        self.output_fields["potential"] = OutputField(self.gfu_w, "potential", VOL)

        deform = self._solver.mesh.prev_deformation[-1]

        # Creating GridFunctions for the Fields
        b_gfu = GridFunction(fes_vector)
        rhs_u_gfu = GridFunction(_fes)
        rhs_w_gfu = GridFunction(_fes)
        grad_phase_bnd_gfu = GridFunction(fes_vector)
        grad_potential_bnd_gfu = GridFunction(fes_vector)
        self.input_fields["b"] = InputField(b_gfu, CF((0,)*solver.mesh.dim), "b", self._solver.ngsmesh.Materials('.*'))
        self.input_fields["rhs_u"] = InputField(rhs_u_gfu, CF(0), "rhs_u", self._solver.ngsmesh.Materials('.*'))
        self.input_fields["rhs_w"] = InputField(rhs_w_gfu, CF(0), "rhs_w", self._solver.ngsmesh.Materials('.*'))
        self.input_fields["grad_phase_bnd"] = InputField(grad_phase_bnd_gfu, CF((0,)*self._solver.ngsmesh.dim), "grad_phase_bnd", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["grad_potential_bnd"] = InputField(grad_potential_bnd_gfu, CF((0,)*self._solver.ngsmesh.dim), "grad_potential_bnd", self._solver.ngsmesh.Boundaries('.*'))

        def dW(phase):
            return (4*phase**3-6*phase**2+2*phase)/4

        def ddW(phase):
            return (12*phase**2-12*phase+2)/4

        self.A += 1/self._solver.time.dt*trial_u*test_u*dx(deformation = deform)
        self.A += b_gfu*grad(trial_u)*test_u*dx(deformation = deform)
        self.A += self.input_params["M"]*grad(trial_w)*grad(test_u)*dx(deformation = deform)
        self.A += trial_w*test_w*dx(deformation = deform)
        self.A += -self.input_params["sigma"]*self.input_params["epsilon"]*grad(trial_u)*grad(test_w)*dx(deformation = deform)
        self.A += -self.input_params["sigma"]/self.input_params["epsilon"]*ddW(self.gfu_u_old)*trial_u*test_w*dx(deformation = deform)

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = fes.FreeDofs())

        self.F += rhs_u_gfu*test_u*dx(deformation = deform)
        self.F += rhs_w_gfu*test_w*dx(deformation = deform)

        self.F += 1/self._solver.time.dt*self.gfu_u_old*test_u*dx(deformation = deform)
        self.F += self.input_params["sigma"]/self.input_params["epsilon"]*dW(self.gfu_u_old)*test_w*dx(deformation = deform)
        self.F += -1*self.input_params["sigma"]/self.input_params["epsilon"]*ddW(self.gfu_u_old)*self.gfu_u_old*test_w*dx(deformation = deform)

        if self.input_params["Neu_bnd_phase"]:
            self.F += -1*self.input_params["sigma"]*self.input_params["epsilon"]*grad_phase_bnd_gfu*ns*test_w*ds(definedon = self.input_params["Neu_bnd_phase"], deformation = deform)
        if self.input_params["Neu_bnd_potential"]:
            self.F += self.input_params["M"]*grad_potential_bnd_gfu*ns*test_u*ds(definedon = self.input_params["Neu_bnd_potential"], deformation = deform)

        if self.input_params["mass_preserving"]:
            ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ir_tet = IntegrationRule(points  = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], \
                                     weights = [1/24, 1/24, 1/24, 1/24])
            dx_lumped = dx(deformation = deform, intrules = { TRIG : ir_trig , TET : ir_tet})
            self.Amp = BilinearForm(self.gfu_u.space, symmetric = True)
            u, v = self.gfu_u.space.TnT()
            self.Amp += u*v*dx_lumped
            self.Amp.Assemble()
            rows,cols,vals = self.Amp.mat.COO()
            self.weights = sp.csr_matrix((vals,(rows,cols))).diagonal()

    def PreProcess(self):
        
        self.gfu_old.vec.data = self.gfu.vec.data

        if self._solver.time.iter == 0:

            if self.input_params["mass_preserving"] and self.input_params["fes_order"]>1:
                raise Exception('Mass preservation not yet implemented for fes_order>1')
            if self.input_params["bounds"] and self.input_params["fes_order"]>1:
                raise Exception('Bounds preservation not yet implemented for fes_order>1')
            
            if self.input_params["bounds"]:
                gfu_vec = self.gfu_u.vec.Copy().FV().NumPy()
                gfu_new = MandBP(gfu_vec, BP = self.input_params["bounds"])
                self.gfu_u.vec.data = gfu_new
        
            if self.input_params["mass_preserving"]:
                gfu0_vec = self.gfu_u.vec.Copy().FV().NumPy()
                self.mass0 = np.sum(self.weights*gfu0_vec)

    def Solve(self):

        self.update_input_fields()

        self.A.Assemble()
        self.invA.Update()
        self.F.Assemble()

        self.gfu.vec.data = self.invA*self.F.vec
        
        if self.input_params["bounds"] and not self.input_params["mass_preserving"]:

            gfu_vec = self.gfu_u.vec.Copy().FV().NumPy()
            gfu_new = MandBP(gfu_vec, BP = self.input_params["bounds"])
            self.gfu_u.vec.data = gfu_new

        elif self.input_params["mass_preserving"]:

            if hasattr(self._solver.time, 'dt'):
                dt = self._solver.time.dt.Get()
            else:
                logger.error('A time-dependent simulation is needed to impose conservative mass')

            self.Amp.Assemble()
            rows,cols,vals = self.Amp.mat.COO()
            weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
            gfu_vec = self.gfu_u.vec.Copy().FV().NumPy()

            if self.input_params["bounds"]:
                BP = self.input_params["bounds"]
            else:
                BP = [-np.inf, np.inf]

            gfu_new = MandBP(gfu_vec, weights=weights, BP=BP,
                                MP=self.input_params["mass_preserving"], mass0=self.mass0, dt = dt)

            self.gfu_u.vec.data = gfu_new

    def PostProcess(self):
        
        pass

    @property
    def phase(self):
        return self.gfu_u
    
    @phase.setter
    def phase(self, cf):
        self.gfu_u.Set(cf, definedon = self.domain)

    @property
    def potential(self):
        return self.gfu_w
    
    @potential.setter
    def potential(self, cf):
        self.gfu_w.Set(cf, definedon = self.domain)

        