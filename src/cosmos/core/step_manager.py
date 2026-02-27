import logging
logger = logging.getLogger(__name__)

import os
import time
from ngsolve import *
from typing import Dict, TYPE_CHECKING
import traceback
import numpy as np
from ngsolve.webgui import Draw

if TYPE_CHECKING:
    from cosmos.core.model import CosmosModel

class CosmosStepManager:

    def __init__(self, kwargs):

        self.subiter = None
        self.params = kwargs
        self.coupling_type = 'explicit'
        self.step_elasped_time = None
        self.adaptive = False

    def initialize(self, model: "CosmosModel"):

        if 'coupling_type' in  self.params.keys():
            if {self.params['coupling_type']} <= {'implicit', 'explicit', 'imex'}:
                self.coupling_type = self.params['coupling_type']
            else:
                raise Exception('Coupling type must be either implicit or explicit')
            
        if 'adaptive_timestep' in  self.params.keys():
            if self.params['adaptive_timestep']:
                self.adaptive = True
            
        for pde in model.pdes:
            pde.Initialize()

        self.len = 0
        self.positions = []
        for i, pde in enumerate(model.pdes_pre):
            self.positions.append(self.len)
            self.len += len(pde.gfu.vec)
        for j, pde in enumerate(model.pdes_post):
            self.positions.append(self.len)
            self.len += len(pde.gfu.vec)
        self.positions.append(self.len)

    def solve_step(self, model: "CosmosModel"):

        start = time.time()

        if len(model.pdes)>0:

            for pde in model.pdes:
                pde.PreProcess()
            for pde in model.pdes_init:
                pde.Solve()

            if self.adaptive:

                print('adaptive')
                print(self.subiter, model.time.dt.Get(), model.time.helper.params['dt'])

                dt_tol = 1e-9

                while model.dt.Get()>dt_tol:
                    
                    try:
                        if self.coupling_type == 'implicit':
                            self.implicit_solve_step_gauss(model)
                        elif self.coupling_type == 'explicit':
                            self.explicit_solve_step(model)
                        elif self.coupling_type == 'imex':
                            self.imex_solve_step_gauss(model)
                        break

                    except:
                        dt = model.dt.Get()
                        model.time.modify_dt(dt/2)

                        for pde in model.pdes:
                            pde.reset()
                        model.ale.reset()
                        print('Timesep reduced: ', model.dt.Get())

                        print(self.subiter, model.time.dt.Get(), model.time.helper.params['dt'])

                if self.subiter<=5:
                    model.time.helper.params['dt'] = (model.time.dt.Get() + model.time.helper.dt0)/2
                else:
                    model.time.helper.params['dt'] = model.time.dt.Get()

                print(self.subiter, model.time.dt.Get(), model.time.helper.params['dt'])

                print('adaptive_end')

                if model.dt.Get()<dt_tol:
                    raise Exception('Timestep shrinked to 0!')
                
            else:

                if self.coupling_type == 'implicit':
                    self.implicit_solve_step_gauss(model)
                elif self.coupling_type == 'explicit':
                    self.explicit_solve_step(model)
                elif self.coupling_type == 'imex':
                    self.imex_solve_step_gauss(model)
                
            logger.debug(f'Subiter solved successfully')
            print('ok')

            for pde in model.pdes:
                pde.PostProcess()

        stop = time.time()
        self.step_elasped_time = stop-start

    def explicit_solve_step(self, model):

        for i, pde in enumerate(model.pdes_pre):
            pde.Solve()
        model.ale.solve_ale(model)
        for j, pde in enumerate(model.pdes_post):
            pde.Solve()

    def implicit_solve_step_gauss(self, model):

        old_sol = []
        tol = 1e-8

        for i, pde in enumerate(model.pdes_pre):
            pde.Solve()
            old_sol.append(pde.gfu.vec.Copy())
        model.ale.solve_ale(model)
        for j, pde in enumerate(model.pdes_post):
            pde.Solve()
            old_sol.append(pde.gfu.vec.Copy())

        errors = np.ones(len(model.pdes_pre) + len(model.pdes_post))*1e5
        self.subiter = 0
        max_iter = 10
        while np.max(errors)>tol and self.subiter < max_iter:

            count = 0
            for i, pde in enumerate(model.pdes_pre):
                pde.Solve()
                errors[count] = Norm(pde.gfu.vec-old_sol[count])/np.max([len(pde.gfu.vec), Norm(old_sol[count])])
                old_sol[count] = pde.gfu.vec.Copy()
                count += 1
            model.ale.solve_ale(model)
            for j, pde in enumerate(model.pdes_post):
                pde.Solve()
                errors[count] = Norm(pde.gfu.vec-old_sol[count])/np.max([len(pde.gfu.vec), Norm(old_sol[count])])
                old_sol[count] = pde.gfu.vec.Copy()
                count += 1
            self.subiter += 1
            logger.debug(f'Step subiter_bool count: {self.subiter} | Max error {np.max(np.array(errors)):.2e}')

            print(errors)

        if self.subiter == max_iter:
            raise Exception(f'Internal solver iteration exceeded max number of {max_iter:d} iterations')
        
    def imex_solve_step_gauss(self, model):
        
        old_sol = []
        tol = 1e-10

        for i, pde in enumerate(model.pdes_pre):
            pde.Solve()
            old_sol.append(pde.gfu.vec.Copy())
        model.ale.solve_ale(model)

        errors = np.ones(len(model.pdes_pre))*1e5

        errors_ale = np.ones(3)*1e5
        old_sol_ale = [model.ale.V.vec.Copy(), model.ale.W.vec.Copy(), model.ale.Y.vec.Copy()]

        max_iter = 10
        self.subiter = 0
        while np.max(errors)>tol and self.subiter < max_iter:

            count = 0
            for i, pde in enumerate(model.pdes_pre):
                pde.Solve()
                errors[count] = Norm(pde.gfu.vec-old_sol[count])/np.max([len(pde.gfu.vec), Norm(old_sol[count])])
                old_sol[count] = pde.gfu.vec.Copy()
                count += 1
            model.ale.solve_ale(model)

            errors_ale[0] = Norm(model.ale.V.vec-old_sol_ale[0])/np.max([len(model.ale.V.vec), Norm(old_sol_ale[0])])
            errors_ale[1] = Norm(model.ale.W.vec-old_sol_ale[1])/np.max([len(model.ale.W.vec), Norm(old_sol_ale[1])])
            errors_ale[2] = Norm(model.ale.Y.vec-old_sol_ale[2])/np.max([len(model.ale.Y.vec), Norm(old_sol_ale[2])])
            old_sol_ale[0] = model.ale.V.vec.Copy()
            old_sol_ale[1] = model.ale.W.vec.Copy()
            old_sol_ale[2] = model.ale.Y.vec.Copy()

            self.subiter += 1
            logger.debug(f'Step subiter_bool count: {self.subiter} | Max error {np.max(np.array(errors)):.2e}')

            print(errors, errors_ale)

        if self.subiter == max_iter:
            print('Max iteration number for nonlinear Gauss iteration reached')
            raise Exception(f'Internal solver iteration exceeded max number of {max_iter:d} iterations')
        
        for j, pde in enumerate(model.pdes_post):
            pde.Solve()

    def implicit_solve_step_anderson(self, model):

        uo = np.zeros(self.len)
        u = np.zeros(self.len)
        f_pic = np.zeros(self.len)
        f_new = np.zeros(self.len)

        count = 0
        for i, pde in enumerate(model.pdes_pre):
            uo[self.positions[count]:self.positions[count+1]] = pde.gfu.vec.FV().NumPy()
            pde.Solve()
            u[self.positions[count]:self.positions[count+1]] = pde.gfu.vec.FV().NumPy()
            count += 1
        model.ale.solve_ale(model)
        for j, pde in enumerate(model.pdes_post):
            uo[self.positions[count]:self.positions[count+1]] = pde.gfu.vec.FV().NumPy()
            pde.Solve()
            u[self.positions[count]:self.positions[count+1]] = pde.gfu.vec.FV().NumPy()
            count += 1
        f = u - uo

        verbose = True
        m=3
        maxit=20
        tol=1e-10
        beta=1.0
        reg=1e-12

        # History of deltas: Δu_i = u_{i+1} - u_i, Δf_i = f_{i+1} - f_i
        dU = []
        dF = []

        norm_u0 = max(np.linalg.norm(u), 1.0)
        rel = np.linalg.norm(f) / norm_u0
        if verbose:
            print(f"it=0  ||f||/||u||={rel:.3e}")

        for k in range(1, maxit + 1):
            if rel < tol:
                break

            # Plain Picard step candidate
            u_pic = u + beta * f
            count = 0
            for i, pde in enumerate(model.pdes_pre):
                pde.gfu.vec.data[:] = u_pic[self.positions[count]:self.positions[count+1]]
                pde.Solve()
                f_pic[self.positions[count]:self.positions[count+1]] = pde.gfu.vec.FV().NumPy() - u_pic[self.positions[count]:self.positions[count+1]]
                count += 1
            model.ale.solve_ale(model)
            for j, pde in enumerate(model.pdes_post):
                pde.gfu.vec.data[:] = u_pic[self.positions[count]:self.positions[count+1]]
                pde.Solve()
                f_pic[self.positions[count]:self.positions[count+1]] = pde.gfu.vec.FV().NumPy() - u_pic[self.positions[count]:self.positions[count +1]]
                count += 1

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
                count = 0
                for i, pde in enumerate(model.pdes_pre):
                    pde.gfu.vec.data[:] = u_new[self.positions[count]:self.positions[count+1]]
                    pde.Solve()
                    f_new[self.positions[count]:self.positions[count+1]] = pde.gfu.vec.FV().NumPy() - u_new[self.positions[count]:self.positions[count+1]]
                    count += 1
                model.ale.solve_ale(model)
                for j, pde in enumerate(model.pdes_post):
                    pde.gfu.vec.data[:] = u_new[self.positions[count]:self.positions[count +1]]
                    pde.Solve()
                    f_new[self.positions[count]:self.positions[count+1]] = pde.gfu.vec.FV().NumPy() - u_new[self.positions[count]:self.positions[count+1]]

                u, f = u_new, f_new

            rel = np.linalg.norm(f) / max(np.linalg.norm(u), 1.0)
            if verbose:
                print(f"it={k}  ||f||/||u||={rel:.3e}  hist={len(dF)}")

        if k >= maxit:
            raise Exception('Exceeded maximum number of iterations') 
        
    def imex_solve_step_anderson(self, model):
        
        uo = np.zeros(self.len)
        u = np.zeros(self.len)
        f_pic = np.zeros(self.len)
        f_new = np.zeros(self.len)

        count = 0
        for i, pde in enumerate(model.pdes_pre):
            uo[self.positions[count]:self.positions[count+1]] = pde.gfu.vec.FV().NumPy()
            pde.Solve()
            u[self.positions[count]:self.positions[count+1]] = pde.gfu.vec.FV().NumPy()
            count += 1
        model.ale.solve_ale(model)
        f = u - uo

        verbose = True
        m=5
        maxit=10
        tol=1e-10
        beta=1.0
        reg=1e-12

        # History of deltas: Δu_i = u_{i+1} - u_i, Δf_i = f_{i+1} - f_i
        dU = []
        dF = []

        norm_u0 = max(np.linalg.norm(u), 1.0)
        rel = np.linalg.norm(f) / norm_u0
        if verbose:
            print(f"it=0  ||f||/||u||={rel:.3e}")

        for k in range(1, maxit + 1):
            if rel < tol:
                break

            # Plain Picard step candidate
            u_pic = u + beta * f
            count = 0
            for i, pde in enumerate(model.pdes_pre):
                pde.gfu.vec.data[:] = u_pic[self.positions[count]:self.positions[count+1]]
                pde.Solve()
                f_pic[self.positions[count]:self.positions[count+1]] = pde.gfu.vec.FV().NumPy() - u_pic[self.positions[count]:self.positions[count+1]]
                count += 1
            model.ale.solve_ale(model)

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
                count = 0
                for i, pde in enumerate(model.pdes_pre):
                    pde.gfu.vec.data[:] = u_new[self.positions[count]:self.positions[count+1]]
                    pde.Solve()
                    f_new[self.positions[count]:self.positions[count+1]] = pde.gfu.vec.FV().NumPy() - u_new[self.positions[count]:self.positions[count+1]]
                    count += 1
                model.ale.solve_ale(model)

                u, f = u_new, f_new

            rel = np.linalg.norm(f) / max(np.linalg.norm(u), 1.0)
            if verbose:
                print(f"it={k}  ||f||/||u||={rel:.3e}  hist={len(dF)}")

        if k >= maxit:
            raise Exception('Exceeded maximum number of iterations') 
        
        print('Post-step')
        
        for j, pde in enumerate(model.pdes_post):
            pde.Solve()