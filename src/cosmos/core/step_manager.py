import logging

logger = logging.getLogger(__name__)

import time
from ngsolve import *
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from cosmos.core.model import CosmosModel


class CosmosStepManager:
    """Orchestrates the per-time-step solve sequence for a CosmosModel.

    Dispatches the Initialize, PreProcess, Solve, and PostProcess phases to all
    registered PDE models in the correct order (pre-ALE PDEs → ALE solve →
    post-ALE PDEs). Supports explicit coupling and implicit (Gauss–Seidel
    iteration) coupling modes, with optional adaptive time-stepping driven by a
    relative solution-change tolerance.
    """

    def __init__(self, model: "CosmosModel", kwargs):

        self.model = model
        self.subiter = None
        self.params = kwargs
        self.coupling_type = "explicit"
        self.step_elasped_time = None
        self.adaptive = False

    def initialize(self):

        if "coupling_type" in self.params.keys():
            if {self.params["coupling_type"]} <= {"implicit", "explicit"}:
                self.coupling_type = self.params["coupling_type"]
            else:
                raise ValueError("Coupling type must be either implicit or explicit")

        if "adaptive_timestep" in self.params.keys():
            if self.params["adaptive_timestep"]:
                self.adaptive = True

        for pde in self.model.pdes:
            pde.Initialize()

        self.len = 0
        self.positions = []
        for i, pde in enumerate(self.model.pdes_pre):
            self.positions.append(self.len)
            self.len += len(pde.gfu.vec)
        for j, pde in enumerate(self.model.pdes_post):
            self.positions.append(self.len)
            self.len += len(pde.gfu.vec)
        self.positions.append(self.len)

        ns = specialcf.normal(self.model.dim)
        Ps = Id(self.model.dim) - OuterProduct(ns, ns)
        V = VectorH1(self.model.parentmesh, definedon=self.model.parentmesh.Boundaries(".*"))
        kappa, xsi = V.TnT()
        self.A = BilinearForm(V)
        self.A += InnerProduct(kappa, xsi) * ds(deformation=self.model.ale.Y)
        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs=V.FreeDofs())
        self.F = LinearForm(V)
        self.F += -1 * InnerProduct(Ps, Grad(xsi).Trace()) * ds(deformation=self.model.ale.Y)
        self.kappa_h = GridFunction(V)

    def solve_step(self):

        start = time.time()

        if len(self.model.pdes) > 0:
            for pde in self.model.pdes:
                pde.PreProcess()
            for pde in self.model.pdes_init:
                pde.Solve()

            if self.adaptive:
                dt_tol = 1e-9

                while self.model.dt.Get() > dt_tol:
                    if self.coupling_type == "implicit":

                        eps_target = 1e-7
                        eps_max = 5e-7
                        eps_min = 5e-8
                        iter_max = 10

                        if eps_max < eps_target or eps_target < eps_min:
                            raise ValueError("Wrong parameters for adaptive algorithm!!")

                        success, eps, subiter = self.implicit_solve_step_gauss(
                            iter_max, eps_min
                        )

                        if eps > eps_max:
                            old_dt = self.model.dt.Get()
                            dt = old_dt * 0.6
                            self.model.time.modify_dt(dt)
                            for pde in self.model.pdes:
                                pde.reset()
                            self.model.ale.reset()
                            logger.info(
                                f"[Cosmos] Adaptive timestep: error {eps:.2e} > "
                                f"tolerance {eps_max:.2e}, reducing dt {old_dt:.4g} -> {dt:.4g}"
                            )
                        elif eps < eps_min:
                            old_dt = self.model.dt.Get()
                            dt = min(old_dt * 1.25, self.model.time.dt0)

                            self.model.time.helper.params["dt"] = dt
                            logger.info(
                                f"[Cosmos] Adaptive timestep: error {eps:.2e} < "
                                f"tolerance {eps_min:.2e}, increasing dt {old_dt:.4g} -> {dt:.4g}"
                            )
                            break
                        else:
                            dt = self.model.dt.Get()
                            self.model.time.helper.params["dt"] = dt
                            logger.debug(
                                f"[Cosmos] Adaptive timestep: error {eps:.2e} within "
                                f"tolerance, keeping dt {dt:.4g}"
                            )
                            break

                    elif self.coupling_type == "explicit":
                        raise NotImplementedError(
                            "No adaptivity implemented for explicit time stepping"
                        )

                if self.model.dt.Get() < dt_tol:
                    logger.error(
                        f"[Cosmos] Adaptive timestep collapsed below tolerance "
                        f"({dt_tol:.1e}) after last error {eps:.2e}; aborting."
                    )
                    raise RuntimeError("Timestep shrinked to 0!")

            else:
                if self.coupling_type == "implicit":
                    eps_max = 5e-7
                    eps_min = 5e-8
                    iter_max = 15
                    success, eps, subiter = self.implicit_solve_step_gauss(iter_max, eps_min)
                    if not success:
                        raise RuntimeError("Implicit algorithm couldn't converge, max_iter reached")
                elif self.coupling_type == "explicit":
                    self.explicit_solve_step()

            logger.debug(f"Subiter solved successfully")

            for pde in self.model.pdes:
                pde.PostProcess()

        stop = time.time()
        self.step_elasped_time = stop - start

    def explicit_solve_step(self):

        for i, pde in enumerate(self.model.pdes_pre):
            pde.Solve()
        self.model.ale.solve_ale()
        for j, pde in enumerate(self.model.pdes_post):
            pde.Solve()

    def implicit_solve_step_gauss(self, iter_max, eps_min):

        old_sol = []
        tol_floor = 1e-12

        if tol_floor > eps_min:
            raise ValueError("The minimum error threshold is too close to machine precision")

        for i, pde in enumerate(self.model.pdes_pre):
            pde.Solve()
            old_sol.append(pde.gfu.vec.Copy())
        self.model.ale.solve_ale()
        old_sol_ale = self.model.ale.Y.vec.Copy()
        for j, pde in enumerate(self.model.pdes_post):
            pde.Solve()
            old_sol.append(pde.gfu.vec.Copy())

        error_ale = 1e5
        errors_pde = np.ones(len(old_sol)) * 1e100
        eps = 1e5

        subiter = 1
        while subiter < iter_max and eps > eps_min:
            count = 0
            for i, pde in enumerate(self.model.pdes_pre):
                pde.Solve()
                errors_pde[count] = Norm(pde.gfu.vec - old_sol[count]) / np.max(
                    [tol_floor, Norm(old_sol[count])]
                )
                old_sol[count].data = pde.gfu.vec.data
                count += 1
            self.model.ale.solve_ale()
            error_ale = Norm(self.model.ale.Y.vec - old_sol_ale) / np.max(
                [tol_floor, Norm(old_sol_ale)]
            )
            old_sol_ale.data = self.model.ale.Y.vec.data
            for j, pde in enumerate(self.model.pdes_post):
                pde.Solve()
                errors_pde[count] = Norm(pde.gfu.vec - old_sol[count]) / np.max(
                    [tol_floor, Norm(old_sol[count])]
                )
                old_sol[count].data = pde.gfu.vec.data
                count += 1
            subiter += 1
            logger.debug(f"Step subiter_bool count: {subiter} | Max error {error_ale:.2e}")
            eps = error_ale
            eps = np.max([np.max(errors_pde), error_ale])

        if subiter == iter_max:
            return False, eps, subiter
        else:
            return True, eps, subiter
