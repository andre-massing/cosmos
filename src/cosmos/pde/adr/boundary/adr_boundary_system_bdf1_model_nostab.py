import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.field import OutputField, Field
from cosmos.core.compartment import CosmosCompartment
from cosmos.core.model import CosmosModel
from cosmos.core.utils import MandBP
from ngsolve.webgui import Draw
import numbers
import time

class ADRBoundarySystemBDF1Model(BasePDEModel):

    def __init__(self, name:str = 'ADRBoundarySystemBDF1Model', model:CosmosModel = None, compartment:CosmosCompartment = None, **kwargs):
        
        super().__init__(name=name, model = model, compartment=compartment)
        
        self.is_bnd = True
        self.is_vol = False
        self.name = name
        self.model = model
        self.compartment = compartment

        self.nonlinearities = []

        if 'dim' in kwargs.keys():
            if isinstance(kwargs['dim'], numbers.Number):
                self.sys_dim = kwargs['dim']
            else:
                raise Exception(f'Dimension of systems {self.name} must be a number')
        else:
            raise Exception('Parameter dim is needed for initialization of ADRBoundarySystemBDF1Model')
        
        self.params["Neu_bnd"] = ''
        self.params["Dir_bnd"] = ''
        self.params["subdivision"] = 0
        self.params["fes_order"] = 1
        self.params["conservative"] = False

        self.V = H1(model.parentmesh, order = self.params["fes_order"], 
                                definedon = compartment.domain)
            
        self.fes = self.V
        for i in range(self.sys_dim-1):
            self.fes = self.fes*self.V

        V2 = VectorH1(model.parentmesh, order = self.params["fes_order"], 
                                definedon = compartment.domain)

        for i in range(self.sys_dim):
            self.params["mass_preserving_" + str(i+1)] = False
            self.params["bounds_" + str(i+1)] = None
            self.params["u0_" + str(i+1)] = CF(0)

            self.params["b_" + str(i+1)] = Field(CF((0,)*compartment.dim_emd))
            self.params["d_" + str(i+1)] = Field(CF(0))
            self.params["c_" + str(i+1)] = Field(CF(0))
            self.params["rhs_" + str(i+1)] = Field(CF(0))
            self.params["gradu_bnd_" + str(i+1)] = Field(CF((0,)*compartment.dim_emd))
            self.params["u_bnd_" + str(i+1)] = Field(CF(0))

            self.params["gfu_b_" + str(i+1)] = GridFunction(V2)
            self.params["gfu_d_" + str(i+1)] = GridFunction(self.V)
            self.params["gfu_c_" + str(i+1)] = GridFunction(self.V)
            self.params["gfu_rhs_" + str(i+1)] = GridFunction(self.V)
            self.params["gfu_gradu_bnd_" + str(i+1)] = GridFunction(V2)
            self.params["gfu_u_bnd_" + str(i+1)] = GridFunction(self.V)
        
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)

        if self.sys_dim>1:
            self.sol = self.gfu.components
        else:
            self.sol = [self.gfu]

        if self.model.dim == 2:
            self.vtk_gfu = [GridFunction(H1(self.model.parentmesh, order = self.params["fes_order"])) for i in range(self.sys_dim)]
        else:
            self.vtk_gfu = self.sol

        ir_segm = IntegrationRule(points = [(0,0), (1,0)], weights = [1/2, 1/2])
        ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
        self.mass0 = np.zeros(self.sys_dim)
        ds_lumped = ds(intrules = {  SEGM : ir_segm, TRIG : ir_trig }, deformation = self.model.ale.Y)
        space = H1(model.parentmesh, order = self.params["fes_order"], 
                                definedon = compartment.domain)
        self.Amp = BilinearForm(space, symmetric = True)
        u, v = space.TnT()
        self.Amp += u*v*ds_lumped
        self.Amp.Assemble()
        rows,cols,vals = self.Amp.mat.COO()
        self.weights0 = sp.csr_matrix((vals,(rows,cols))).diagonal()

        self.vtk_names = [self.name + '_c' + str(i+1) for i in range(self.sys_dim)]

        self.Yhalf = GridFunction(model.ale.Y.space)

    def Initialize(self):

        for i in range(self.sys_dim):
            self.sol[i].Set(self.params["u0_" + str(i+1)], definedon = self.compartment.domain, dual = True)

            if self.params["mass_preserving_" + str(i+1)] and self.params["fes_order"]>1:
                raise Exception('Mass preservation not yet implemented for fes_order>1')
            if self.params["bounds_" + str(i+1)] and self.params["fes_order"]>1:
                raise Exception('Bounds preservation not yet implemented for fes_order>1')
            
            if self.params["mass_preserving_" + str(i+1)]:
                gfu0_vec = self.output_fields["sol_" + str(i+1)]._coef.vec.Copy().FV().NumPy()
                self.mass0[i] = np.sum(self.weights0*gfu0_vec)
            
            if self.params["bounds_" + str(i+1)]:
                self.sol[i].vec.data[:] = np.clip(self.sol[i].vec.FV().NumPy(), 
                        self.params["bounds_" + str(i+1)][0], self.params["bounds_" + str(i+1)][1])
                
        if self.model.dim == 2:
            for i, gfu in enumerate(self.vtk_gfu):
                gfu.Set(self.sol[i], definedon = self.compartment.domain)

        if self.sys_dim>1:
            trial, test = self.fes.TnT()
        else:
            trial = [self.fes.TrialFunction()]
            test = [self.fes.TestFunction()]
        self.A = BilinearForm(self.fes)
        self.F = LinearForm(self.fes)
        self.ns = specialcf.normal(self.model.dim)
        Ps = Id(self.model.dim) - OuterProduct(self.ns, self.ns)
        h = self.cfg.h
        alpha = 5 * self.params["fes_order"] * (self.params["fes_order"]+1)
        tE = specialcf.tangential(self.model.dim)
        if self.model.dim == 2:
            facet_space = self.V
            nE = specialcf.tangential(self.model.dim)
        else:
            facet_space = FacetSurface(self.model.parentmesh, order = 0)
            nE = Cross(self.ns, tE)

        for i in range(self.sys_dim):

            self.A += self.params["gfu_c_" + str(i+1)]*trial[i]*test[i]*ds(deformation = self.Yhalf)
            self.A += self.params["gfu_d_" + str(i+1)]*grad(trial[i]).Trace()*grad(test[i]).Trace()*ds(deformation = self.Yhalf)
                    
            if self.params['Dir_bnd']:
                dir_bnd_gfu = GridFunction(facet_space)
                dir_bnd_gfu.Set(1, definedon = self.model.parentmesh.BBoundaries(self.params['Dir_bnd']))
                self.A += - dir_bnd_gfu*self.params["gfu_d_" + str(i+1)]*InnerProduct(nE, grad(trial[i]).Trace())*test[i]*ds(element_boundary=True, deformation = self.Yhalf) \
                    - dir_bnd_gfu*self.params["gfu_d_" + str(i+1)]*InnerProduct(nE, grad(test[i]).Trace())*trial[i]*ds(element_boundary=True, deformation = self.Yhalf)\
                    + dir_bnd_gfu*self.params["gfu_d_" + str(i+1)]*alpha/h*trial[i]*test[i]*ds(element_boundary=True, deformation = self.Yhalf)

            self.A += -(self.params["gfu_b_" + str(i+1)])*grad(test[i]).Trace() * trial[i]*ds(deformation = self.Yhalf)
            bnd_gfu = GridFunction(facet_space)
            bnd_gfu.Set(1, definedon = self.model.parentmesh.BBoundaries(self.params['Dir_bnd']+'|'+self.params['Neu_bnd']))
            self.A += bnd_gfu*IfPos((self.params["gfu_b_" + str(i+1)])*nE, (self.params["gfu_b_" + str(i+1)])*nE*trial[i], CF(0))*test[i]\
                *ds(element_boundary=True, deformation = self.Yhalf)

            self.F += self.params["gfu_rhs_" + str(i+1)]*test[i]*ds(deformation = self.Yhalf)
            
            if self.params['Dir_bnd']:
                self.F += dir_bnd_gfu*self.params["gfu_d_" + str(i+1)]*alpha/h*self.params["gfu_u_bnd_" + str(i+1)]*test[i]*ds(element_boundary=True, deformation = self.Yhalf)\
                    - dir_bnd_gfu*self.params["gfu_d_" + str(i+1)]*InnerProduct(nE, grad(test[i]).Trace())*self.params["gfu_u_bnd_" + str(i+1)]*ds(element_boundary=True, deformation = self.Yhalf)
            if self.params['Neu_bnd']:
                neu_bnd_gfu = GridFunction(facet_space)
                neu_bnd_gfu.Set(1, definedon = self.model.parentmesh.BBoundaries(self.params['Neu_bnd']))
                self.F += neu_bnd_gfu*self.params["gfu_d_" + str(i+1)]*self.params["gfu_gradu_bnd_" + str(i+1)]*nE*test[i]*ds(element_boundary=True, deformation = self.Yhalf)

            self.F += -bnd_gfu*IfPos((self.params["gfu_b_" + str(i+1)])*nE, CF(0), (self.params["gfu_b_" + str(i+1)])*nE*self.params["gfu_u_bnd_" + str(i+1)])*test[i]*ds(element_boundary=True, deformation = self.Yhalf)
            
            self.A += 1/self.model.dt*trial[i]*test[i]*ds(deformation = self.model.ale.Y)

            if self.sys_dim > 1:
                self.F += 1/self.model.dt*self.gfu_old.components[i]*test[i]*ds(deformation = self.model.ale.Yo)
            else:
                self.F += 1/self.model.dt*self.gfu_old*test[0]*ds(deformation = self.model.ale.Yo)
            
        for nonlin in self.nonlinearities:

            env = {"__builtins__": {}}
            env.update(self._base_env())
            for j in range(self.sys_dim):
                env.update({"u"+str(j+1): self.sol[j], "v"+str(j+1): test[j]})
            env.update(nonlin['map'])
            
            self.F += -1*eval(nonlin['expr'], env)*test[(nonlin['target']-1)]*ds(deformation = self.Yhalf)

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = self.fes.FreeDofs())

    def PreProcess(self):
        
        self.gfu_old.vec.data = self.gfu.vec.data

    def Solve(self):

        self.Yhalf.vec.data = 0.5*self.model.ale.Y.vec.data + 0.5*self.model.ale.Yo.vec.data

        ############### Do I actually need this? Getting rid of it for now
        ############### might be a problem for manufactured solutions
        self.model.time.helper.t.Set(self.model.time.t.Get() + 0.5*self.model.time.dt.Get())

        for i in range(self.sys_dim):

            self.params["gfu_c_" + str(i+1)].Set(self.params["c_" + str(i+1)]().Compile(), definedon = self.compartment.domain)
            self.params["gfu_d_" + str(i+1)].Set(self.params["d_" + str(i+1)]().Compile(), definedon = self.compartment.domain)
            self.params["gfu_b_" + str(i+1)].Set(self.params["b_" + str(i+1)]().Compile() - self.model.ale.Wo, definedon = self.compartment.domain)
            self.params["gfu_rhs_" + str(i+1)].Set(self.params["rhs_" + str(i+1)]().Compile(), definedon = self.compartment.domain)
            self.params["gfu_u_bnd_" + str(i+1)].Set(self.params["u_bnd_" + str(i+1)]().Compile(), definedon = self.compartment.domain)
            self.params["gfu_gradu_bnd_" + str(i+1)].Set(self.params["gradu_bnd_" + str(i+1)]().Compile(), definedon = self.compartment.domain)

        self.A.Assemble()
        self.invA.Update()

        # if self.nonlinearities:
        
        #     verbose = True
        #     m=5
        #     maxit=20
        #     tol=1e-10
        #     beta=1.0
        #     reg=1e-12
            
        #     u = self.gfu.vec.Copy().FV().NumPy()

        #     # History of deltas: Δu_i = u_{i+1} - u_i, Δf_i = f_{i+1} - f_i
        #     dU = []
        #     dF = []

        #     # Initial evaluation
        #     self.F.Assemble()
        #     self.gfu.vec.data = self.invA*self.F.vec
        #     f = self.gfu.vec.FV().NumPy() - u

        #     norm_u0 = max(np.linalg.norm(u), 1.0)
        #     rel = np.linalg.norm(f) / norm_u0
        #     if verbose:
        #         print(f"it=0  ||f||/||u||={rel:.3e}")

        #     for k in range(1, maxit + 1):
        #         if rel < tol:
        #             break

        #         # Plain Picard step candidate
        #         u_pic = u + beta * f
        #         self.gfu.vec.data[:] = u_pic
        #         self.F.Assemble()
        #         self.gfu.vec.data = self.invA*self.F.vec
        #         f_pic = self.gfu.vec.FV().NumPy() - u_pic

        #         # Update histories with newest step information
        #         # (use u_pic and f_pic as the "next" quantities)
        #         du = (u_pic - u)
        #         df = (f_pic - f)

        #         if np.linalg.norm(df) > 0:
        #             dU.append(du)
        #             dF.append(df)
        #             if len(dU) > m:
        #                 dU.pop(0)
        #                 dF.pop(0)

        #         # If not enough history yet, accept Picard
        #         if len(dF) == 0:
        #             u, f = u_pic, f_pic
        #         else:
        #             # Build least squares: minimize || f_pic - DF * gamma ||, DF columns are dF_j
        #             DF = np.column_stack(dF)  # shape (N, p)
        #             # Solve (DF^T DF + reg I) gamma = DF^T f_pic
        #             A = DF.T @ DF
        #             A.flat[::A.shape[0] + 1] += reg  # add reg to diagonal
        #             b = DF.T @ f_pic
        #             gamma = np.linalg.solve(A, b)

        #             # Anderson update:
        #             # u_{new} = u_pic - DU * gamma  (where DU columns are dU_j)
        #             DU = np.column_stack(dU)
        #             u_new = u_pic - DU @ gamma

        #             # Recompute f at accelerated iterate
        #             self.gfu.vec.data[:] = u_new
        #             self.F.Assemble()
        #             self.gfu.vec.data = self.invA*self.F.vec
        #             f_new = self.gfu.vec.FV().NumPy() - u_new

        #             u, f = u_new, f_new

        #         rel = np.linalg.norm(f) / max(np.linalg.norm(u), 1.0)
        #         if verbose:
        #             print(f"it={k}  ||f||/||u||={rel:.3e}  hist={len(dF)}")

        #     if k >= maxit:
        #         raise Exception('Exceeded maximum number of iterations') 
            
        # else:

        self.F.Assemble()
        self.gfu.vec.data = self.invA*self.F.vec

        ############### Do I actually need this? Getting rid of it for now
        ############### might be a problem for manufactured solutions
        self.model.time.helper.t.Set(self.model.time.t.Get())

        self.gfu.vec.data = self.invA*self.F.vec

        for i in range(self.sys_dim):

            if self.params["bounds_" + str(i+1)] and not self.params["mass_preserving_" + str(i+1)]:

                self.Amp.Assemble()
                rows,cols,vals = self.Amp.mat.COO()
                weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
                gfu_vec = self.sol[i].vec.Copy().FV().NumPy()
                
                gfu_new = MandBP(gfu_vec, weights = weights, BP = self.params["bounds_" + str(i+1)], 
                                 MP = True, mass0 = np.sum(weights*gfu_vec), dt = dt)
                self.sol[i].vec.data = gfu_new

            elif self.params["mass_preserving_" + str(i+1)]:

                if hasattr(self.model.time, 'dt'):
                    dt = self.model.dt.Get()
                else:
                    logger.error('A time-dependent simulation is needed to impose conservative mass')

                self.Amp.Assemble()
                rows,cols,vals = self.Amp.mat.COO()
                weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
                gfu_vec = self.sol[i].vec.Copy().FV().NumPy()

                if self.params["bounds_" + str(i+1)]:
                    BP = self.params["bounds_" + str(i+1)]
                else:
                    BP = [-np.inf, np.inf]

                gfu_new = MandBP(gfu_vec, weights=weights, BP=BP,
                                    MP=self.params["mass_preserving_" + str(i+1)], mass0=self.mass0[i], dt = dt)

                self.sol[i].vec.data = gfu_new

    def PostProcess(self):

        if self.model.dim == 2:
            for i, gfu in enumerate(self.vtk_gfu):
                gfu.Set(self.sol[i], definedon = self.compartment.domain)

    def _base_env(self):
        # whitelist of functions (extend as needed)
        return {
            "sin": sin, "cos": cos, "exp": exp, "log": log, "sqrt": sqrt,
            "IfPos": IfPos, "x": x, "y": y, "z": z
        }

    def add_nonlinearity(self, target, expression, map = {}):

        nonlin = {}
        nonlin["expr"] = expression
        nonlin["map"] = map
        nonlin["target"] = target

        self.nonlinearities.append(nonlin)