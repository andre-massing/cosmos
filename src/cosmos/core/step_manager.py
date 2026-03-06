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

        self.control = []

    def initialize(self, model: "CosmosModel"):

        if 'coupling_type' in  self.params.keys():
            if {self.params['coupling_type']} <= {'implicit', 'explicit'}:
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

        ns = specialcf.normal(model.dim)
        Ps = Id(model.dim) - OuterProduct(ns, ns)
        V = VectorH1(model.parentmesh, definedon = model.parentmesh.Boundaries('.*'))
        kappa, xsi = V.TnT()
        self.A = BilinearForm(V)
        self.A += InnerProduct(kappa, xsi)*ds(deformation = model.ale.Y)
        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs = V.FreeDofs())
        self.F = LinearForm(V)
        self.F += -1*InnerProduct(Ps, Grad(xsi).Trace())*ds(deformation = model.ale.Y)
        self.kappa_h = GridFunction(V) 

        self.control.append(self.energy(model))

    def solve_step(self, model: "CosmosModel"):

        start = time.time()

        if len(model.pdes)>0:

            for pde in model.pdes:
                pde.PreProcess()
            for pde in model.pdes_init:
                pde.Solve()

            if self.adaptive:

                safety = 0.8

                print('adaptive')

                dt_tol = 1e-9

                while model.dt.Get()>dt_tol:
                    
                    if self.coupling_type == 'implicit':

                        order = 1
                        iter_target = 4

                        success, subiter = self.implicit_solve_step_gauss(model)

                        if not success:
                            dt = model.dt.Get()
                            dt *= safety * (iter_target / subiter) ** (1 / (order + 1))
                            model.time.modify_dt(dt)
                            for pde in model.pdes:
                                pde.reset()
                            model.ale.reset()
                            print('Control difference is too high, timestep lowered to: ', dt)
                        elif subiter<=iter_target:
                            dt = model.dt.Get()
                            dt *= safety * (iter_target / subiter) ** (1 / (order + 1))
                            dt = min(dt, model.time.dt0)

                            model.time.helper.params['dt'] = dt
                            print('Control difference is too low, timestep raised to: ', dt)
                            break
                        else:
                            dt = model.dt.Get()
                            model.time.helper.params['dt'] = dt
                            break

                    elif self.coupling_type == 'explicit':

                        previous_control = self.control[-1]
                        eps_target = 1e-5
                        eps_max = 1e-4
                        eps_min = 1e-6
                        small_floor = 1e-8
                        order = 1

                        if eps_max<eps_target or eps_target<eps_min:
                            raise Exception('Wrong parameters for adaptive algorithm!!')

                        self.explicit_solve_step(model)

                        new_control = self.energy(model)
                        self.control.append(new_control)
                        eps = np.max([abs((new_control - previous_control)/previous_control), small_floor])

                        print(previous_control, new_control, eps)

                        if eps > eps_max:
                            dt = model.dt.Get()
                            dt *= safety * (eps_target / eps) ** (1 / (order + 1))
                            model.time.modify_dt(dt)
                            for pde in model.pdes:
                                pde.reset()
                            model.ale.reset()

                            print('Control difference is too high, timestep reduced to: ', dt)
                        elif eps < eps_min:
                            dt = model.dt.Get()
                            dt *= safety * (eps_target / eps) ** (1 / (order + 1))
                            dt = min(dt, model.time.dt0)

                            model.time.helper.params['dt'] = dt
                            print('Control difference is too low, timestep raised to: ', dt)
                            break
                        else:
                            dt = model.dt.Get()
                            model.time.helper.params['dt'] = dt
                            print('Control inside limits, timestep kept as: ', dt)
                            break

                if model.dt.Get()<dt_tol:
                    print(self.control)
                    raise Exception('Timestep shrinked to 0!')
                
                print('adaptive_end')
                
            else:

                if self.coupling_type == 'implicit':
                    success, subiter = self.implicit_solve_step_gauss(model)
                    if not success:
                        raise Exception('Implicit algorithm couldn\'t converge, max_iter reached')
                elif self.coupling_type == 'explicit':
                    self.explicit_solve_step(model)
                
            logger.debug(f'Subiter solved successfully')
            print('ok')

            for pde in model.pdes:
                pde.PostProcess()

        stop = time.time()
        self.step_elasped_time = stop-start

    def energy(self, model: "CosmosModel"):

        model.parentmesh.deformation.vec.data = model.ale.Y.vec.data

        self.A.Assemble()
        self.invA.Update()
        self.F.Assemble()

        self.kappa_h.vec.data = self.invA*self.F.vec
        w_energy = Integrate(self.kappa_h**2, model.parentmesh, VOL_or_BND = BND)
        if model.dim == 2:
            v_energy = Integrate(CF((x, 0))*specialcf.normal(model.dim), model.parentmesh, VOL_or_BND = BND)
        elif model.dim == 3:
            v_energy = Integrate(CF((x, 0, 0))*specialcf.normal(model.dim), model.parentmesh, VOL_or_BND = BND)
        a_energy = Integrate(1, model.parentmesh, VOL_or_BND = BND)

        model.parentmesh.deformation.vec.data = model.ale.Yo.vec.data

        return w_energy

    def explicit_solve_step(self, model):

        for i, pde in enumerate(model.pdes_pre):
            pde.Solve()
        model.ale.solve_ale(model)
        for j, pde in enumerate(model.pdes_post):
            pde.Solve()

    def implicit_solve_step_gauss(self, model):

        old_sol = []
        error_ale = 1e5
        old_sol_ale = model.ale.dY.vec.Copy()
        tol = 1e-8

        for i, pde in enumerate(model.pdes_pre):
            pde.Solve()
            old_sol.append(pde.gfu.vec.Copy())
        model.ale.solve_ale(model)
        for j, pde in enumerate(model.pdes_post):
            pde.Solve()
            old_sol.append(pde.gfu.vec.Copy())

        subiter = 0
        max_iter = 10
        while error_ale>tol and subiter < max_iter:

            count = 0
            for i, pde in enumerate(model.pdes_pre):
                pde.Solve()
                count += 1
            model.ale.solve_ale(model)
            error_ale = Norm(model.ale.dY.vec-old_sol_ale)/np.max([len(model.ale.dY.vec), Norm(old_sol_ale)])
            old_sol_ale = model.ale.dY.vec.Copy()
            for j, pde in enumerate(model.pdes_post):
                pde.Solve()
                count += 1
            subiter += 1
            logger.debug(f'Step subiter_bool count: {subiter} | Max error {error_ale:.2e}')

            print('errors_ale', error_ale)

        if subiter == max_iter:
            return False, subiter
        else:
            return True, subiter