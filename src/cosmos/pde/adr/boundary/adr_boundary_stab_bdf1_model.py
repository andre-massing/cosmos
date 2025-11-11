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

class ADRBoundaryStabBDF1Model(BasePDEModel):

    def __init__(self, solver:Solver,
                 model_order:int,
                 domain:str = '.*',
                 name:str = 'ADRBoundaryStabBDF1Model',
                 input_params = {}):
        
        super().__init__()
        
        self.VorB = BND
        self.name = name
        self._solver = solver
        self.model_order = model_order

        if (domain in solver.mesh.bnd_markers) or domain == '.*':
            self.domain = solver.ngsmesh.Boundaries(domain)
        else:
            logger.error('The domain specified for the Model ', self.name, ' does not exist')

        solver._attach_model(self, self.model_order)

        self.input_params["Neu_bnd"] = ''
        self.input_params["Dir_bnd"] = ''
        self.input_params["periodic"] = False
        self.input_params["mass_preserving"] = False
        self.input_params["bounds"] = None
        self.input_params["fes_order"] = 1
        self.input_params["u0"] = CF(0)

        self.set_input_params(input_params)

        V = H1(self._solver.ngsmesh, order = self.input_params["fes_order"], 
                                            definedon = self.domain)
        if self._solver.ngsmesh.dim == 2:
            dV = H1(self._solver.ngsmesh, order = self.input_params["fes_order"],
                definedon=self.domain)
        else:
            dV = NormalFacetSurface(self._solver.ngsmesh, order = self.input_params["fes_order"]-1,
                definedon=self.domain)
        
        if self.input_params["periodic"]:
            fes = CompressCompound(Periodic(V)*Periodic(dV))
            fes_vector = Compress(Periodic(VectorH1(self._solver.ngsmesh, order = self.input_params["fes_order"], 
                                            definedon = self.domain)))
        else:
            fes = CompressCompound(V*dV)
            fes_vector = Compress(VectorH1(self._solver.ngsmesh, order = self.input_params["fes_order"], 
                                   definedon = self.domain))
            
        n = specialcf.normal(self._solver.mesh.dim)
        Ps = Id(self._solver.ngsmesh.dim) - OuterProduct(n, n)
        h = self.cfg.h
        alpha = 5 * self.input_params["fes_order"] * (self.input_params["fes_order"]+1)
        tE = specialcf.tangential(self._solver.ngsmesh.dim)
        if self._solver.ngsmesh.dim == 2:
            facet_space = V
            nE = specialcf.tangential(self._solver.ngsmesh.dim)
            tEc = CF((-n[1], n[0]))
        else:
            facet_space = FacetSurface(self._solver.ngsmesh, order = 0)
            nE = Cross(n, tE)
        bnd_gfu = GridFunction(facet_space)
        bnd_gfu.Set(1, definedon = self._solver.ngsmesh.BBoundaries(self.input_params['Dir_bnd']+'|'+self.input_params['Neu_bnd']))
          
        (trial, trial_d), (test, test_d) = fes.TnT()
        self.A = BilinearForm(fes)
        self.F = LinearForm(fes)
        
        self.gfu = GridFunction(fes)
        self.gfu_sol, self.gfu_sol_d = self.gfu.components
        self.gfu_old = GridFunction(fes)
        gfu_sol_old = self.gfu_old.components[0]
        self.gfu_sol.Set(self.input_params['u0'], definedon = self.domain, dual = True)
        self.output_fields["sol"] = OutputField(self.gfu_sol, "sol", BND)

        deform = self._solver.mesh.curr_deformation
        deform_old = self._solver.mesh.prev_deformation[-1]

        # Creating GridFunctions for the Fields
        b_gfu = GridFunction(fes_vector)
        d_gfu = GridFunction(V)
        c_gfu = GridFunction(V)
        rhs_gfu = GridFunction(V)
        gradu_gfu = GridFunction(fes_vector)
        u_bnd_gfu = GridFunction(V)
        self.input_fields["b"] = InputField(b_gfu, CF((0,)*solver.mesh.dim), "b", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["d"] = InputField(d_gfu, CF(0), "d", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["c"] = InputField(c_gfu, CF(0), "c", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["rhs"] = InputField(rhs_gfu, CF(0), "rhs", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["gradu_bnd"] = InputField(gradu_gfu, CF((0,)*solver.mesh.dim), "gradu_bnd", self._solver.ngsmesh.Boundaries('.*'))
        self.input_fields["u_bnd"] = InputField(u_bnd_gfu, CF(0), "u_bnd", self._solver.ngsmesh.Boundaries('.*'))

        if self.input_params["mass_preserving"]:
            ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            ds_lumped = ds(intrules = {  SEGM : ir_segm, TRIG : ir_trig }, deformation = deform)
            self.Amp = BilinearForm(self.gfu_sol.space, symmetric = True)
            u, v = self.gfu_sol.space.TnT()
            self.Amp += u*v*ds_lumped
            self.Amp.Assemble()
            rows,cols,vals = self.Amp.mat.COO()
            self.weights = sp.csr_matrix((vals,(rows,cols))).diagonal()

        self.A += c_gfu*trial*test*ds(deformation = deform)
        self.A += d_gfu*grad(trial).Trace()*grad(test).Trace()*ds(deformation = deform)
                
        if self.input_params['Dir_bnd']:
            dir_bnd_gfu = GridFunction(facet_space)
            dir_bnd_gfu.Set(1, definedon = self._solver.ngsmesh.BBoundaries(self.input_params['Dir_bnd']))
            self.A += - dir_bnd_gfu*d_gfu*InnerProduct(nE, grad(trial).Trace())*test*ds(element_boundary=True, deformation = deform) \
                - dir_bnd_gfu*d_gfu*InnerProduct(nE, grad(test).Trace())*trial*ds(element_boundary=True, deformation = deform)\
                + dir_bnd_gfu*d_gfu*alpha/h*trial*test*ds(element_boundary=True, deformation = deform)\

        self.A += -b_gfu*grad(test).Trace() * trial*ds(deformation = deform)
        self.A += bnd_gfu*IfPos(b_gfu*nE, b_gfu*nE*trial, CF(0))*test\
            *ds(element_boundary=True, deformation = deform)
        
        if self._solver.ngsmesh.dim == 2:
            tEc = CF((-n[1], n[0]))
            jump_dudn = (trial.Trace().Deriv() - trial_d*tEc)*nE
            jump_dvdn = (test.Trace().Deriv() - test_d*tEc)*nE
        elif self._solver.ngsmesh.dim == 3:
            jump_dudn = (trial.Trace().Deriv() - trial_d.Trace())*nE
            jump_dvdn = (test.Trace().Deriv() - test_d.Trace())*nE
        stab = Norm(b_gfu)*h**2
        epsilon = 1e-8
        if self._solver.ngsmesh.dim == 2:
            b_gfu.Set(CF((epsilon, epsilon)), definedon = self.domain)
        elif self._solver.ngsmesh.dim == 3:
            b_gfu.Set(CF((epsilon, epsilon, epsilon)), definedon = self.domain)
        self.A +=  stab*InnerProduct(jump_dudn,jump_dvdn)\
                *ds(element_boundary=True, deformation = deform)
        self.A +=  -1*stab*bnd_gfu*InnerProduct(jump_dudn,jump_dvdn)\
                *ds(element_boundary=True, deformation = deform)
        self.A +=  bnd_gfu*InnerProduct(trial_d.Trace(),test_d.Trace())\
                *ds(element_boundary=True, deformation = deform)
        
        self.A += 1/self._solver.time.dt*trial*test*ds(deformation = deform)

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = fes.FreeDofs())

        self.F += rhs_gfu*test*ds(deformation = deform)
        
        if self.input_params['Dir_bnd']:
            self.F += dir_bnd_gfu*d_gfu*alpha/h*u_bnd_gfu*test*ds(element_boundary=True, deformation = deform)\
                - dir_bnd_gfu*d_gfu*InnerProduct(nE, grad(test).Trace())*u_bnd_gfu*ds(element_boundary=True, deformation = deform)
        if self.input_params['Neu_bnd']:
            neu_bnd_gfu = GridFunction(facet_space)
            neu_bnd_gfu.Set(1, definedon = self._solver.ngsmesh.BBoundaries(self.input_params['Neu_bnd']))
            self.F += neu_bnd_gfu*d_gfu*gradu_gfu*nE*test*ds(element_boundary=True, deformation = deform)

        self.F += -bnd_gfu*IfPos(b_gfu*nE, CF(0), b_gfu*nE*u_bnd_gfu)*test*ds(element_boundary=True, deformation = deform)

        self.F += 1/self._solver.time.dt*gfu_sol_old*test*ds(deformation = deform_old)

    def PreProcess(self):
        
        self.gfu_old.vec.data = self.gfu.vec.data

        if self._solver.time.iter == 0:

            if self.input_params["mass_preserving"] and self.input_params["fes_order"]>1:
                raise Exception('Mass preservation not yet implemented for fes_order>1')
            if self.input_params["bounds"] and self.input_params["fes_order"]>1:
                raise Exception('Bounds preservation not yet implemented for fes_order>1')
        
            if self.input_params["mass_preserving"]:
                gfu0_vec = self.gfu_sol.vec.Copy().FV().NumPy()
                self.mass0 = np.sum(self.weights*gfu0_vec)


    def Solve(self):
        
        self._solver.time.advance_tcoef()
        self._solver.mesh.advance_mesh()
        self.update_input_fields()

        self.A.Assemble()
        self.invA.Update()
        self.F.Assemble()

        self.gfu.vec.data = self.invA*self.F.vec
        
        if self.input_params["bounds"] and not self.input_params["mass_preserving"]:

            gfu_vec = self.gfu_sol.vec.Copy().FV().NumPy()
            gfu_new = MandBP(gfu_vec, BP = self.input_params["bounds"])
            self.gfu_sol.vec.data = gfu_new

        elif self.input_params["mass_preserving"]:

            if hasattr(self._solver.time, 'dt'):
                dt = self._solver.time.dt.Get()
            else:
                logger.error('A time-dependent simulation is needed to impose conservative mass')

            self.Amp.Assemble()
            rows,cols,vals = self.Amp.mat.COO()
            weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
            gfu_vec = self.gfu_sol.vec.Copy().FV().NumPy()

            if self.input_params["bounds"]:
                BP = self.input_params["bounds"]
            else:
                BP = [-np.inf, np.inf]

            gfu_new = MandBP(gfu_vec, weights=weights, BP=BP,
                                MP=self.input_params["mass_preserving"], mass0=self.mass0, dt = dt)

            self.gfu_sol.vec.data = gfu_new

        self._solver.time.reset_tcoef()
        self._solver.mesh.reset_mesh()

    def PostProcess(self):
        
        pass

    @property
    def sol(self):
        return self.gfu_sol
    
    @sol.setter
    def sol(self, cf):
        self.gfu_sol.Set(cf, definedon = self.domain)

        