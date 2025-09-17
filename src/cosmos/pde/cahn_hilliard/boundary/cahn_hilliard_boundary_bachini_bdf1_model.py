import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.solver import Solver
from cosmos.core.field import InputField, OutputField
from cosmos.core.utils import MandBP

class CahnHilliardBoundaryBachiniBDF1Model(BasePDEModel):

    def __init__(self, solver:Solver,
                 model_order:int,
                 domain:str = '.*',
                 name:str = 'CahnHilliardBoundaryBachiniBDF1Model',
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

        n = specialcf.normal(self._solver.ngsmesh.dim)
        tE = specialcf.tangential(self._solver.ngsmesh.dim)
        Ps = Id(self._solver.ngsmesh.dim) - OuterProduct(n, n)
        if self._solver.ngsmesh.dim == 2:
            facet_space = H1(self._solver.ngsmesh, order = 1, 
                                            definedon = self.domain)
            nE = specialcf.tangential(self._solver.ngsmesh.dim)
        else:
            facet_space = FacetSurface(self._solver.ngsmesh, order = 0)
            nE = Cross(n, tE)

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
        self.gfu_u.Set(self.input_params['u0'], dual = True, definedon = self.domain)
        self.output_fields["phase"] = OutputField(self.gfu_u, "phase", BND)
        self.gfu_w.Set(self.input_params['w0'], dual = True, definedon = self.domain)
        self.output_fields["potential"] = OutputField(self.gfu_w, "potential", BND)

        deform = self._solver.mesh.prev_deformation[-1]

        # Creating GridFunctions for the Fields
        b_gfu = GridFunction(fes_vector)
        rhs_u_gfu = GridFunction(_fes)
        rhs_w_gfu = GridFunction(_fes)
        grad_phase_bnd_gfu = GridFunction(fes_vector)
        grad_potential_bnd_gfu = GridFunction(fes_vector)
        self.input_fields["b"] = InputField(b_gfu, CF((0,)*solver.mesh.dim), "b", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["rhs_u"] = InputField(rhs_u_gfu, CF(0), "rhs_u", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["rhs_w"] = InputField(rhs_w_gfu, CF(0), "rhs_w", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["grad_phase_bnd"] = InputField(grad_phase_bnd_gfu, CF((0,)*self._solver.ngsmesh.dim), "grad_phase_bnd", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["grad_potential_bnd"] = InputField(grad_potential_bnd_gfu, CF((0,)*self._solver.ngsmesh.dim), "grad_potential_bnd", self._solver.ngsmesh.Boundaries('.*'))

        def dW(phase):
            return phase**3-phase

        def ddW(phase):
            return 3*phase**2-1

        self.A += 1/self._solver.time.dt*trial_u*test_u*ds(deformation = deform)
        self.A += b_gfu*grad(trial_u).Trace()*test_u*ds(deformation = deform)
        self.A += self.input_params["M"]*grad(trial_w).Trace()*grad(test_u).Trace()*ds(deformation = deform)
        self.A += trial_w*test_w*ds(deformation = deform)
        # self.A += -self.input_params["epsilon"]*grad(trial_u).Trace()*grad(test_w).Trace()*ds(deformation = deform)
        self.A += -self.input_params["sigma"]*self.input_params["epsilon"]*grad(trial_u).Trace()*grad(test_w).Trace()*ds(deformation = deform)
        self.A += -self.input_params["sigma"]/self.input_params["epsilon"]*ddW(self.gfu_u_old)*trial_u*test_w*ds(deformation = deform)

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = fes.FreeDofs())

        self.F += rhs_u_gfu*test_u*ds(deformation = deform)
        self.F += rhs_w_gfu*test_w*ds(deformation = deform)
        
        self.F += 1/self._solver.time.dt*self.gfu_u_old*test_u*ds(deformation = deform)
        self.F += self.input_params["sigma"]/self.input_params["epsilon"]*dW(self.gfu_u_old)*test_w*ds(deformation = deform)
        self.F += -1*self.input_params["sigma"]/self.input_params["epsilon"]*ddW(self.gfu_u_old)*self.gfu_u_old*test_w*ds(deformation = deform)

        if self.input_params["Neu_bnd_phase"]:
            neu_bnd_gfu_phase = GridFunction(facet_space)
            neu_bnd_gfu_phase.Set(1, definedon = self._solver.ngsmesh.BBoundaries(self.input_params['Neu_bnd_phase']))
            self.F += -1*neu_bnd_gfu_phase*self.input_params["sigma"]*self.input_params["epsilon"]*grad_phase_bnd_gfu*nE*test_w*ds(element_boundary = True, deformation = deform)
        if self.input_params["Neu_bnd_potential"]:
            neu_bnd_gfu_potential = GridFunction(facet_space)
            neu_bnd_gfu_potential.Set(1, definedon = self._solver.ngsmesh.BBoundaries(self.input_params['Neu_bnd_potential']))
            self.F += neu_bnd_gfu_potential*self.input_params["M"]*grad_potential_bnd_gfu*nE*test_u*ds(element_boundary=True, deformation = deform)

        if self.input_params["mass_preserving"]:
            ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = {  SEGM : ir_segm, TRIG : ir_trig })
            self.Amp = BilinearForm(self.gfu_u.space, symmetric = True)
            u, v = self.gfu_u.space.TnT()
            self.Amp += u*v*ds_lumped
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

    @property
    def energy(self):
        energy = Integrate(self.input_params['sigma']/self.input_params['epsilon']*0.25*(self.phase**2-1)**2
                           +self.input_params['sigma']*self.input_params['epsilon']/2*Norm(grad(self.phase).Trace())**2, 
                           self._solver.ngsmesh, VOL_or_BND = BND)
        return energy

        