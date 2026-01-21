import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.field import InputField, OutputField, Field
from cosmos.core.compartment import CosmosCompartment
from cosmos.core.model import CosmosModel
from cosmos.core.utils import MandBP
from ngsolve.webgui import Draw
import numbers
import time
from ngsolve.solvers import *

class ADRVolumeSystemBDF1Model(BasePDEModel):

    def __init__(self, name:str = 'ADRVolumeSystemBDF1Model', model:CosmosModel = None, compartment:CosmosCompartment = None, **kwargs):
        
        super().__init__(name=name, model = model, compartment=compartment)
        
        self.is_bnd = False
        self.is_vol = True
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
        self.params["fes_order"] = 1
        self.params["subdivision"] = 0
        self.params["conservative"] = False
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
            self.params["tot_flux_bnd_" + str(i+1)] = Field(CF(0))

        V = H1(model.parentmesh, order = self.params["fes_order"], 
                                definedon = compartment.domain, dgjumps = True)
        self.fes = V
        for i in range(self.sys_dim-1):
            self.fes = self.fes*V
        
        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)
        if self.sys_dim>1:
            self.sol = self.gfu.components
            for i in range(self.sys_dim):
                self.output_fields["sol_" + str(i+1)] = OutputField(self.gfu.components[i], "sol_" + str(i+1), VOL)
        else:
            self.sol = [self.gfu]
            self.output_fields["sol_1"] = OutputField(self.gfu, "sol_1", VOL)

        self.vectorspace = VectorH1(model.parentmesh, order = 1, definedon = compartment.domain)
        self.Amap_h = GridFunction(self.vectorspace)
        self.discscalarspace = L2(model.parentmesh, order = 0, definedon =compartment.domain)
        self.J_h_old = GridFunction(self.discscalarspace)
        if model.dim == 2:
            self.identity = CF((x,y))
        else:
            self.identity = CF((x,y,z))

        self.dX_new = GridFunction(self.model.dX.space)
        self.dX_new = GridFunction(self.model.dX.space)
        self.dX_old = GridFunction(self.model.dX.space)
        self.mass0 = np.zeros(self.sys_dim)
        ir_trig = IntegrationRule(points = [(0,0), (1,0), (0,1)], weights = [1/6, 1/6, 1/6])
        ir_tet = IntegrationRule(points  = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], \
                                        weights = [1/24, 1/24, 1/24, 1/24])
        dx_lumped = dx(intrules = {  TRIG : ir_trig , TET : ir_tet }, deformation = self.dX_new)
        space = H1(model.parentmesh, order = self.params["fes_order"], 
                                definedon = compartment.domain)
        self.Amp = BilinearForm(space, symmetric = True)
        u, v = space.TnT()
        self.Amp += u*v*dx_lumped
        self.Amp.Assemble()
        rows,cols,vals = self.Amp.mat.COO()
        self.weights0 = sp.csr_matrix((vals,(rows,cols))).diagonal()

    def Initialize(self):

        for i in range(self.sys_dim):
            self.output_fields["sol_" + str(i+1)]._coef.Set(self.params["u0_" + str(i+1)], definedon = self.compartment.domain, dual = True)

            if self.params["mass_preserving_" + str(i+1)] and self.params["fes_order"]>1:
                raise Exception('Mass preservation not yet implemented for fes_order>1')
            if self.params["bounds_" + str(i+1)] and self.params["fes_order"]>1:
                raise Exception('Bounds preservation not yet implemented for fes_order>1')
            
            if self.params["mass_preserving_" + str(i+1)]:
                gfu0_vec = self.sol[i].vec.Copy().FV().NumPy()
                self.mass0[i] = np.sum(self.weights0*gfu0_vec)
            
            if self.params["bounds_" + str(i+1)]:
                self.output_fields["sol_" + str(i+1)]._coef.vec.data[:] = np.clip(self.output_fields["sol_" + str(i+1)]._coef.vec.FV().NumPy(),
                                                                                  self.params["bounds_" + str(i+1)][0], self.params["bounds_" + str(i+1)][1])

        if self.params['printing']:
            output_vtk_folder = os.path.join(self.model.io.root, self.name)
            output_vtk_name = os.path.join(output_vtk_folder, self.name)
            os.makedirs(output_vtk_folder, exist_ok=True)
            gfu_one = GridFunction(H1(self.model.parentmesh, definedon = self.compartment.domain))
            gfu_one.Set(1, definedon = self.compartment.domain)
            self.vtk = VTKOutput(self.model.parentmesh,
                                coefs=[self.output_fields["sol_" + str(i+1)]._coef for i in range(self.sys_dim)] + [gfu_one],
                                names =['concentration_' + str(i+1) for i in range(self.sys_dim)]+ ['indicator'],
                                filename= output_vtk_name, 
                                subdivision = self.params['subdivision'])

    def PreProcess(self):
        
        self.gfu_old.vec.data = self.gfu.vec.data

        vectorV_h_old = self.model.ale.ale_velocity
        dt = self.model.dt.Get()
        # self.Amap_h.Set(self.identity - dt*vectorV_h_old, dual = True, definedon =self.compartment.domain)
        # self.J_h_old.Set(Det(Grad(self.Amap_h).trans*Grad(self.Amap_h)), definedon=self.compartment.domain)

        self.dX_new.vec.data = self.model.dX.vec.data
        self.dX_old.vec.data = self.model.ale.prev_dX[-2].data

    def Solve(self):

        if self.sys_dim>1:
            trial, test = self.fes.TnT()
        else:
            trial = [self.fes.TrialFunction()]
            test = [self.fes.TestFunction()]
        self.A = BilinearForm(self.fes)
        self.F = LinearForm(self.fes)

        n = specialcf.normal(self.model.dim)
        h = self.cfg.h
        alpha = 5 * self.params["fes_order"] * (self.params["fes_order"]+1)

        ############### Do I actually need this? Getting rid of it for now
        ############### might be a problem for manufactured solutions
        # self.model.time.helper.t.Set(self.model.time.t.Get() + self.model.time.dt.Get())

        for i in range(self.sys_dim):

            c = self.params["c_" + str(i+1)]().Compile()
            d = self.params["d_" + str(i+1)]().Compile()
            b = self.params["b_" + str(i+1)]().Compile()
            rhs = self.params["rhs_" + str(i+1)]().Compile()
            u_bnd = self.params["u_bnd_" + str(i+1)]().Compile()
            gradu_bnd = self.params["gradu_bnd_" + str(i+1)]().Compile()
            tot_flux_bnd = self.params["tot_flux_bnd_" + str(i+1)]().Compile()

            self.A += c*trial[i]*test[i]*dx(deformation = self.dX_new)
            self.A += d*grad(trial[i])*grad(test[i])*dx(deformation = self.dX_new)
                    
            if self.params['Dir_bnd']:
                self.A += - d*InnerProduct(n, grad(trial[i]))*test[i]*ds(definedon = self.params['Dir_bnd'], skeleton=True, deformation = self.dX_new) \
                    - d*InnerProduct(n, grad(test[i]))*trial[i]*ds(definedon = self.params['Dir_bnd'], skeleton=True, deformation = self.dX_new)\
                    + d*alpha/h*trial[i]*test[i]*ds(definedon = self.params['Dir_bnd'], skeleton = True, deformation = self.dX_new)\

            self.A += -b*grad(test[i]) * trial[i]*dx(deformation = self.dX_new)
            stab = (Norm(b))*h**2
            jump_u = grad(trial[i])-grad(trial[i]).Other()
            jump_v = grad(test[i])-grad(test[i]).Other()
            self.A += stab*jump_u*jump_v*dx(deformation = self.dX_new, skeleton = True)
            self.A += IfPos(b*n, b*n*trial[i], CF(0))*test[i]\
                *ds(deformation = self.dX_new)

            self.F += rhs*test[i]*dx(deformation = self.dX_new)
            
            if self.params['Dir_bnd']:
                self.F += d*alpha/h*u_bnd*test[i]*ds(definedon = self.params['Dir_bnd'], skeleton = True, deformation = self.dX_new)\
                    - d*InnerProduct(n, grad(test[i]))*u_bnd*ds(definedon = self.params['Dir_bnd'], skeleton=True, deformation = self.dX_new)
            if self.params['Neu_bnd']:
                self.F += d*gradu_bnd*n*test[i]*ds(definedon = self.params['Neu_bnd'], deformation = self.dX_new)

            self.F += tot_flux_bnd*test[i]*ds(deformation = self.dX_new)

            self.F += -IfPos(b*n, CF(0), b*n*u_bnd)*test[i]*ds(deformation = self.dX_new)

            self.A += 1/self.model.dt*trial[i]*test[i]*dx(deformation = self.dX_new)

            if self.sys_dim > 1:
                self.F += 1/self.model.dt*self.gfu_old.components[i]*test[i]*dx(deformation = self.dX_old)
            elif self.sys_dim == 1:
                self.F += 1/self.model.dt*self.gfu_old*test[0]*dx(deformation = self.dX_old)

        for nonlin in self.nonlinearities:

            env = {"__builtins__": {}}
            env.update(self._base_env())
            for j in range(self.sys_dim):
                env.update({"u"+str(j+1): self.sol[j], "v"+str(j+1): test[j]})
            env.update(nonlin['map'])
            
            self.F += -1*eval(nonlin['expr'], env)*test[nonlin['target']-1]*dx(deformation = self.dX_new)

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = self.fes.FreeDofs())

        verbose = True
        m=5
        maxit=50
        tol=1e-10
        beta=1.0
        reg=1e-12
        
        u = self.gfu.vec.Copy().FV().NumPy()

        # History of deltas: Δu_i = u_{i+1} - u_i, Δf_i = f_{i+1} - f_i
        dU = []
        dF = []

        # Initial evaluation
        self.F.Assemble()
        self.gfu.vec.data = self.invA*self.F.vec
        f = self.gfu.vec.FV().NumPy() - u

        norm_u0 = max(np.linalg.norm(u), 1.0)
        rel = np.linalg.norm(f) / norm_u0
        if verbose:
            print(f"it=0  ||f||/||u||={rel:.3e}")

        for k in range(1, maxit + 1):
            if rel < tol:
                break

            # Plain Picard step candidate
            u_pic = u + beta * f
            self.gfu.vec.data[:] = u_pic
            self.F.Assemble()
            self.gfu.vec.data = self.invA*self.F.vec
            f_pic = self.gfu.vec.FV().NumPy() - u_pic

            # Update histories with newest step information
            # (use u_pic and f_pic as the "next" quantities)
            du = (u_pic - u)
            df = (f_pic - f)

            if np.linalg.norm(df) > 0:
                dU.append(du)
                dF.append(df)
                if len(dU) > m:
                    dU.pop(0)
                    dF.pop(0)

            # If not enough history yet, accept Picard
            if len(dF) == 0:
                u, f = u_pic, f_pic
            else:
                # Build least squares: minimize || f_pic - DF * gamma ||, DF columns are dF_j
                DF = np.column_stack(dF)  # shape (N, p)
                # Solve (DF^T DF + reg I) gamma = DF^T f_pic
                A = DF.T @ DF
                A.flat[::A.shape[0] + 1] += reg  # add reg to diagonal
                b = DF.T @ f_pic
                gamma = np.linalg.solve(A, b)

                # Anderson update:
                # u_{new} = u_pic - DU * gamma  (where DU columns are dU_j)
                DU = np.column_stack(dU)
                u_new = u_pic - DU @ gamma

                # Recompute f at accelerated iterate
                self.gfu.vec.data[:] = u_new
                self.F.Assemble()
                self.gfu.vec.data = self.invA*self.F.vec
                f_new = self.gfu.vec.FV().NumPy() - u_new

                u, f = u_new, f_new

            rel = np.linalg.norm(f) / max(np.linalg.norm(u), 1.0)
            if verbose:
                print(f"it={k}  ||f||/||u||={rel:.3e}  hist={len(dF)}")

        if k >= maxit:
            raise Exception('Exceeded maximum number of iterations') 

        ############## Do I actually need this? Getting rid of it for now
        ############## might be a problem for manufactured solutions
        # self.model.time.helper.t.Set(self.model.time.t.Get())

        for i in range(self.sys_dim):

            if self.params["bounds_" + str(i+1)] and not self.params["mass_preserving_" + str(i+1)]:

                gfu_vec = self.output_fields["sol_" + str(i+1)]._coef.vec.Copy().FV().NumPy()
                gfu_new = MandBP(gfu_vec, BP = self.params["bounds_" + str(i+1)])
                self.output_fields["sol_" + str(i+1)]._coef.vec.data = gfu_new

            elif self.params["mass_preserving_" + str(i+1)]:

                if hasattr(self.model.time, 'dt'):
                    dt = self.model.dt.Get()
                else:
                    logger.error('A time-dependent simulation is needed to impose conservative mass')

                self.Amp.Assemble()
                rows,cols,vals = self.Amp.mat.COO()
                weights = sp.csr_matrix((vals,(rows,cols))).diagonal()
                gfu_vec = self.output_fields["sol_" + str(i+1)]._coef.vec.Copy().FV().NumPy()

                if self.params["bounds_" + str(i+1)]:
                    BP = self.params["bounds_" + str(i+1)]
                else:
                    BP = [-np.inf, np.inf]

                gfu_new = MandBP(gfu_vec, weights=weights, BP=BP,
                                    MP=self.params["mass_preserving_" + str(i+1)], mass0=self.mass0[i], dt = dt)

                self.output_fields["sol_" + str(i+1)]._coef.vec.data = gfu_new

    def PostProcess(self):

        del self.A
        del self.invA
        del self.F

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
