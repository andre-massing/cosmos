import logging
logger = logging.getLogger(__name__)

import numpy as np
from ngsolve import *
from cosmos.config.parameters import get_config
from cosmos.pde.base import BasePDEModel
from cosmos.core.field import InputField, OutputField, Field
from cosmos.core.solver import Solver
from ngsolve.webgui import Draw
import numbers

class ALEField(Field):

    def __init__(self, solver:Solver, coef: CoefficientFunction, domain:str, VorB,  clamped_bnd:str='', redistribute = False):

        super().__init__(coef)
        
        self._solver = solver
        self.dims = self().dims
        self.name = domain
        self.domain = domain
        self.clamped_bnd = clamped_bnd
        self.redistribute = redistribute
        self.VorB = VorB

        if self.VorB == BND:
            V = Compress(VectorH1(self._solver.ngsmesh, order = self._solver.ngsmesh.GetCurveOrder(), 
                         definedon = self._solver.ngsmesh.Boundaries(domain),
                         dirichlet_bbnd = self._solver.ngsmesh.BBoundaries(clamped_bnd)))
            self.ale_displ = GridFunction(V)
        elif self.VorB == VOL:
            V = Compress(VectorH1(self._solver.ngsmesh, order = self._solver.ngsmesh.GetCurveOrder(), 
                         definedon = self._solver.ngsmesh.Materials(domain)))
            self.ale_displ = GridFunction(V)

        if self.redistribute:

            deformation0 = GridFunction(self._solver.mesh.curr_deformation.space)
            self.mat_def = GridFunction(self._solver.mesh.curr_deformation.space)

            self.ale_displ = GridFunction(V)
            V2 = Compress(H1(self._solver.ngsmesh, order = self._solver.ngsmesh.GetCurveOrder(),
                    definedon = self._solver.ngsmesh.Boundaries(domain)))

            fes_pp = V*V2
            self.A_pp = BilinearForm(fes_pp, symmetric = True)
            self.F_pp = LinearForm(fes_pp)
            self.duanli = GridFunction(fes_pp)
            self.wind = self.duanli.components[0]

            (w, kappa), (eta, mu) = fes_pp.TnT()
            ns = specialcf.normal(self._solver.ngsmesh.dim)
            Ps = Id(self._solver.ngsmesh.dim) - OuterProduct(ns, ns)

            # ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
            # ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
            
            self.A_pp += (InnerProduct(grad(w).Trace(), grad(eta).Trace()))*ds(deformation = deformation0)
            self.A_pp += (-1*InnerProduct(eta*ns, kappa))*ds(deformation=self.mat_def)
            self.A_pp += (-1*InnerProduct(w*ns, mu))*ds(deformation=self.mat_def)
            self.A_pp.Assemble()
            self.invA_pp = self.A_pp.mat.Inverse(freedofs = fes_pp.FreeDofs())

            self.F_pp += (-1*InnerProduct(Ps, grad(eta).Trace()))*ds(deformation = deformation0)
            self.F_pp += (-1*InnerProduct(grad(self.mat_def).Trace(), grad(eta).Trace()))*ds(deformation = deformation0)
            self.F_pp += (InnerProduct(grad(deformation0).Trace(), grad(eta).Trace()))*ds(deformation = deformation0)

    @property
    def cf(self):
        return self._eval()

    @cf.setter
    def cf(self, new_coef):
        if isinstance(new_coef, GridFunction) or isinstance(new_coef, CoefficientFunction):
            self._coef = new_coef
        elif isinstance(new_coef, numbers.Number):
            self._coef = CF(new_coef)
        elif callable(new_coef):
            self._coef = new_coef
        else:
            raise Exception("Unsupported type for Field " +  self.name)
        
    def update(self):

        if self.VorB == BND:
            self.ale_displ.Set(self(), definedon = self._solver.ngsmesh.Boundaries(self.domain), dual = True)
        elif self.VorB == VOL:
            self.ale_displ.Set(self(), definedon = self.domain, dual = True)

        if self.redistribute:
            self.mat_def.Set(self.ale_displ + self._solver.mesh.prev_deformation[-1], dual=True, definedon=self._solver.ngsmesh.Boundaries(self.domain))
            self.A_pp.Assemble()
            self.invA_pp.Update()
            self.F_pp.Assemble()
            self.duanli.vec.data = self.invA_pp*self.F_pp.vec
            self.ale_displ.vec.data += self.wind.vec.data
        

class ALEModel(BasePDEModel):

    def __init__(self, solver:Solver, model_order:int, name:str ='ALEModel'):
        
        super().__init__()

        self.name = name
        self._solver = solver
        self.model_order = model_order

        solver._attach_model(self, self.model_order)
        self.domain = self._solver.mesh.domain        
        self.gfu = GridFunction(self._solver.mesh.curr_deformation.space)
        self.aux_gfu = GridFunction(self._solver.mesh.curr_deformation.space)
        self.tot_wind = GridFunction(self._solver.mesh.curr_deformation.space)
        self.aux_tot_wind = GridFunction(self._solver.mesh.curr_deformation.space)

        if self._solver.ngsmesh.ne != 0:
            self.output_fields["displacement"] = OutputField(self.gfu, "displacement", VOL)
            vol_space = VectorH1(self._solver.ngsmesh, order=self._solver.ngsmesh.GetCurveOrder(),
                                definedon = self._solver.ngsmesh.Materials('.*'), dirichlet = self._solver.ngsmesh.Boundaries('.*'))
            u, v = vol_space.TnT()
            self.A = BilinearForm(vol_space, symmetric = True)

            self.A += InnerProduct(grad(u), grad(v))*dx
            self.A.Assemble()
            self.invA = self.A.mat.Inverse(freedofs = vol_space.FreeDofs())
        else:
            self.output_fields["displacement"] = OutputField(self.gfu, "displacement", BND)

        self.bnd_fields = {}
        self.bnd_is_prescribed = False
        self.vol_fields = {}
        self.vol_is_prescribed = False

    def PreProcess(self):
        
        pass

    def Solve(self):

        self.gfu.vec.data[:] = 0
        self.tot_wind.vec.data[:] = 0

        if self.bnd_is_prescribed:

            redistribute = False

            for field in self.bnd_fields.values():
                field.update()
                self.aux_gfu.Set(field.ale_displ, definedon = self._solver.ngsmesh.Boundaries(field.domain), dual = True)
                self.gfu.vec.data += self.aux_gfu.vec.data

                if field.redistribute:
                    self.aux_tot_wind.Set(field.wind, definedon = self._solver.ngsmesh.Boundaries(field.domain), dual = True)
                    self.tot_wind.vec.data += -1*self.aux_tot_wind.vec.data
                    redistribute = True

            if self._solver.ngsmesh.ne != 0:
                self.A.Assemble()
                self.invA.Update()
                res = -1*self.A.mat*self.gfu.vec
                self.gfu.vec.data += self.invA*res

                if redistribute:
                    res = -1*self.A.mat*self.tot_wind.vec
                    self.tot_wind.vec.data += self.invA*res

        else:

            for field in self.vol_fields.values():
                field.update()
                self.aux_gfu.Set(field.ale_def, definedon = field.domain)
                self.gfu.vec.data += self.aux_gfu.vec.data

        self._solver.mesh.curr_deformation.vec.data = self._solver.mesh.prev_deformation[-1].vec.data + self.gfu.vec.data

    def PostProcess(self):
        
        self._solver.mesh.update_state()

    def set_bnd_displacement(self, coef, domain, clamped_bnd='', redistribute=False):

        if self.vol_is_prescribed:
            raise Exception('Cannot prescribe boundary displcement if volume displacement is already prescribed')
        else:
            self.bnd_is_prescribed = True

        if not all(word in self._solver.mesh.bnd_markers for word in domain.split('|')):
            raise Exception('Boundary not present in list of boundary for the mesh')
        if clamped_bnd:
            if not all(word in self._solver.mesh.bbnd_markers for word in clamped_bnd.split('|')):
                raise Exception('Clamped boudnary not present in list of boundaries for the mesh')
        
        self.bnd_fields[domain] = ALEField(self._solver, coef, domain, BND, clamped_bnd=clamped_bnd, redistribute=redistribute)

    def set_vol_displacement(self, coef, domain):

        if self.bnd_is_prescribed:
            raise Exception('Cannot prescribe volume displacement if volume displacement is already prescribed')
        else:
            self.vol_is_prescribed = True
        
        if not all(word in self._solver.mesh.vol_markers for word in domain.split('|')) and self._solver.ngsmesh.ne == 0:
            raise Exception('The mesh is a surface mesh, use set_bnd_displacement instead')
        if not all(word in self._solver.mesh.vol_markers for word in domain.split('|')) and not self._solver.ngsmesh.ne == 0:
            raise Exception('Material not present in list of materials for the mesh')
        
        self.vol_fields[domain] = ALEField(self._solver, coef, domain, VOL)

    @property
    def displacement(self):
        return self.gfu
    
    @property
    def ale_vel(self):
        return self.gfu/self._solver.time.dt
    
    @property
    def mat_vel(self):
        return (self.tot_wind+self.gfu)/self._solver.time.dt
    
    @property
    def wind(self):
        return self.tot_wind/self._solver.time.dt