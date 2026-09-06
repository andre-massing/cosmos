import logging

logger = logging.getLogger(__name__)

from typing import Optional

import numpy as np
import scipy.sparse as sp

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.field import Field
from cosmos.core.compartment import CosmosCompartment
from cosmos.core.model import CosmosModel
from cosmos.core.utils import MandBP
import numbers
from ngsolve.solvers import *


class ADRVolumeSystemBDF1Model(BasePDEModel):
    """BDF1 advection–diffusion–reaction system on a volume compartment.

    Discretises a coupled system of ``dim`` scalar species on a volumetric domain
    using H1 finite elements with interior-penalty (SIP) stabilisation and a
    first-order backward-difference time scheme. Supports Dirichlet, Neumann,
    and total-flux boundary conditions, nonlinear reaction terms, mass
    preservation, and bound-preserving post-processing.
    """

    is_bnd = False
    is_vol = True

    def __init__(
        self,
        name: str = "ADRVolumeSystemBDF1Model",
        model: Optional[CosmosModel] = None,
        compartment: Optional[CosmosCompartment] = None,
        **kwargs,
    ):

        super().__init__(name=name, model=model, compartment=compartment)

        self.nonlinearities = []

        if "dim" in kwargs.keys():
            if isinstance(kwargs["dim"], numbers.Number):
                self.sys_dim = kwargs["dim"]
            else:
                raise TypeError(f"Dimension of systems {self.name} must be a number")
        else:
            raise ValueError(
                "Parameter dim is needed for initialization of ADRVolumeSystemBDF1Model"
            )

        self.params["Neu_bnd"] = ""
        self.params["Dir_bnd"] = ""
        self.params["fes_order"] = 1
        self.params["subdivision"] = 0

        V = H1(
            model.parentmesh,
            order=self.params["fes_order"],
            definedon=compartment.domain,
            dgjumps=True,
        )
        self.fes = V
        for i in range(self.sys_dim - 1):
            self.fes = self.fes * V

        V2 = VectorH1(
            model.parentmesh, order=self.params["fes_order"], definedon=compartment.domain
        )

        for i in range(self.sys_dim):
            self.params["mass_preserving_" + str(i + 1)] = False
            self.params["bounds_" + str(i + 1)] = None
            self.params["u0_" + str(i + 1)] = CF(0)

            self.params["b_" + str(i + 1)] = Field(CF((0,) * compartment.dim_emd))
            self.params["d_" + str(i + 1)] = Field(CF(0))
            self.params["c_" + str(i + 1)] = Field(CF(0))
            self.params["rhs_" + str(i + 1)] = Field(CF(0))
            self.params["gradu_bnd_" + str(i + 1)] = Field(CF((0,) * compartment.dim_emd))
            self.params["u_bnd_" + str(i + 1)] = Field(CF(0))
            self.params["tot_flux_bnd_" + str(i + 1)] = Field(CF(0))

            self.params["gfu_b_" + str(i + 1)] = GridFunction(V2)
            self.params["gfu_d_" + str(i + 1)] = GridFunction(V)
            self.params["gfu_c_" + str(i + 1)] = GridFunction(V)
            self.params["gfu_rhs_" + str(i + 1)] = GridFunction(V)
            self.params["gfu_gradu_bnd_" + str(i + 1)] = GridFunction(V2)
            self.params["gfu_u_bnd_" + str(i + 1)] = GridFunction(V)
            self.params["gfu_tot_flux_bnd_" + str(i + 1)] = GridFunction(V)

        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)
        if self.sys_dim > 1:
            self.sol = self.gfu.components
        else:
            self.sol = [self.gfu]

        self.mass0 = np.zeros(self.sys_dim)
        ir_trig = IntegrationRule(points=[(0, 0), (1, 0), (0, 1)], weights=[1 / 6, 1 / 6, 1 / 6])
        ir_tet = IntegrationRule(
            points=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
            weights=[1 / 24, 1 / 24, 1 / 24, 1 / 24],
        )
        dx_lumped = dx(intrules={TRIG: ir_trig, TET: ir_tet}, deformation=self.model.ale.Y)
        space = H1(model.parentmesh, order=self.params["fes_order"], definedon=compartment.domain)
        self.Amp = BilinearForm(space, symmetric=True)
        u, v = space.TnT()
        self.Amp += u * v * dx_lumped
        self.Amp.Assemble()
        rows, cols, vals = self.Amp.mat.COO()
        self.weights0 = sp.csr_matrix((vals, (rows, cols))).diagonal()

        self.vtk_gfu = [self.sol[i] for i in range(self.sys_dim)]
        self.vtk_names = [self.name + "_c" + str(i + 1) for i in range(self.sys_dim)]

        self.Yhalf = GridFunction(model.ale.Y.space)

    def Initialize(self):

        for i in range(self.sys_dim):
            self.sol[i].Set(
                self.params["u0_" + str(i + 1)], definedon=self.compartment.domain, dual=True
            )

            if self.params["mass_preserving_" + str(i + 1)] and self.params["fes_order"] > 1:
                raise NotImplementedError("Mass preservation not yet implemented for fes_order>1")
            if self.params["bounds_" + str(i + 1)] and self.params["fes_order"] > 1:
                raise NotImplementedError("Bounds preservation not yet implemented for fes_order>1")

            if self.params["mass_preserving_" + str(i + 1)]:
                gfu0_vec = self.sol[i].vec.Copy().FV().NumPy()
                self.mass0[i] = np.sum(self.weights0 * gfu0_vec)

            if self.params["bounds_" + str(i + 1)]:
                self.sol[i].vec.data[:] = np.clip(
                    self.sol[i].vec.FV().NumPy(),
                    self.params["bounds_" + str(i + 1)][0],
                    self.params["bounds_" + str(i + 1)][1],
                )

        if self.sys_dim > 1:
            trial, test = self.fes.TnT()
        else:
            trial = [self.fes.TrialFunction()]
            test = [self.fes.TestFunction()]
        self.A = BilinearForm(self.fes)
        self.F = LinearForm(self.fes)

        n = specialcf.normal(self.model.dim)
        h = specialcf.mesh_size
        alpha = 5 * self.params["fes_order"] * (self.params["fes_order"] + 1)

        for i in range(self.sys_dim):
            self.A += (
                self.params["gfu_c_" + str(i + 1)] * trial[i] * test[i] * dx(deformation=self.Yhalf)
            )
            self.A += (
                self.params["gfu_d_" + str(i + 1)]
                * grad(trial[i])
                * grad(test[i])
                * dx(deformation=self.Yhalf)
            )

            if self.params["Dir_bnd"]:
                self.A += (
                    -self.params["gfu_d_" + str(i + 1)]
                    * InnerProduct(n, grad(trial[i]))
                    * test[i]
                    * ds(definedon=self.params["Dir_bnd"], skeleton=True, deformation=self.Yhalf)
                    - self.params["gfu_d_" + str(i + 1)]
                    * InnerProduct(n, grad(test[i]))
                    * trial[i]
                    * ds(definedon=self.params["Dir_bnd"], skeleton=True, deformation=self.Yhalf)
                    + self.params["gfu_d_" + str(i + 1)]
                    * alpha
                    / h
                    * trial[i]
                    * test[i]
                    * ds(definedon=self.params["Dir_bnd"], skeleton=True, deformation=self.Yhalf)
                )

            self.A += (
                -(self.params["gfu_b_" + str(i + 1)])
                * grad(test[i])
                * trial[i]
                * dx(deformation=self.Yhalf)
            )
            jump_u = grad(trial[i]) - grad(trial[i]).Other()
            jump_v = grad(test[i]) - grad(test[i]).Other()
            self.A += (
                (Norm(self.params["gfu_b_" + str(i + 1)]))
                * h**2
                * jump_u
                * jump_v
                * dx(deformation=self.Yhalf, skeleton=True)
            )
            self.A += (
                IfPos(
                    (self.params["gfu_b_" + str(i + 1)]) * n,
                    (self.params["gfu_b_" + str(i + 1)]) * n * trial[i],
                    CF(0),
                )
                * test[i]
                * ds(deformation=self.Yhalf)
            )

            self.F += self.params["gfu_rhs_" + str(i + 1)] * test[i] * dx(deformation=self.Yhalf)

            if self.params["Dir_bnd"]:
                self.F += self.params["gfu_d_" + str(i + 1)] * alpha / h * self.params[
                    "gfu_u_bnd_" + str(i + 1)
                ] * test[i] * ds(
                    definedon=self.params["Dir_bnd"], skeleton=True, deformation=self.Yhalf
                ) - self.params["gfu_d_" + str(i + 1)] * InnerProduct(
                    n, grad(test[i])
                ) * self.params["gfu_u_bnd_" + str(i + 1)] * ds(
                    definedon=self.params["Dir_bnd"], skeleton=True, deformation=self.Yhalf
                )
            if self.params["Neu_bnd"]:
                self.F += (
                    self.params["gfu_d_" + str(i + 1)]
                    * self.params["gfu_gradu_bnd_" + str(i + 1)]
                    * n
                    * test[i]
                    * ds(definedon=self.params["Neu_bnd"], deformation=self.Yhalf)
                )

            self.F += (
                -IfPos(
                    (self.params["gfu_b_" + str(i + 1)]) * n,
                    CF(0),
                    (self.params["gfu_b_" + str(i + 1)])
                    * n
                    * self.params["gfu_u_bnd_" + str(i + 1)],
                )
                * test[i]
                * ds(deformation=self.Yhalf)
            )

            self.A += 1 / self.model.dt * trial[i] * test[i] * dx(deformation=self.model.ale.Y)

            if self.sys_dim > 1:
                self.F += (
                    1
                    / self.model.dt
                    * self.gfu_old.components[i]
                    * test[i]
                    * dx(deformation=self.model.ale.Yo)
                )
            elif self.sys_dim == 1:
                self.F += (
                    1 / self.model.dt * self.gfu_old * test[0] * dx(deformation=self.model.ale.Yo)
                )

        for nonlin in self.nonlinearities:
            env = {"__builtins__": {}}
            env.update(self._base_env())
            if self.sys_dim > 1:
                for j in range(self.sys_dim):
                    env.update(
                        {"u" + str(j + 1): self.gfu_old.components[j], "v" + str(j + 1): test[j]}
                    )
            elif self.sys_dim == 1:
                env.update({"u1": self.gfu_old, "v1": test[0]})
            env.update(nonlin["map"])

            self.F += (
                -1
                * eval(nonlin["expr"], env)
                * test[nonlin["target"] - 1]
                * dx(deformation=self.Yhalf)
            )

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs=self.fes.FreeDofs())

        logger.info(
            f"[Cosmos] PDE '{self.name}' ({type(self).__name__}) initialized "
            f"on compartment '{self.compartment.name}' (dim={self.sys_dim})"
        )

    def PreProcess(self):

        self.gfu_old.vec.data = self.gfu.vec.data

    def Solve(self):

        self.Yhalf.vec.data = 0.5 * self.model.ale.Y.vec.data + 0.5 * self.model.ale.Yo.vec.data

        self.model.time.helper.t.Set(self.model.time.t.Get() + 0.5 * self.model.time.dt.Get())

        for i in range(self.sys_dim):
            self.params["gfu_c_" + str(i + 1)].Set(
                self.params["c_" + str(i + 1)]().Compile(), definedon=self.compartment.domain
            )
            self.params["gfu_d_" + str(i + 1)].Set(
                self.params["d_" + str(i + 1)]().Compile(), definedon=self.compartment.domain
            )
            self.params["gfu_b_" + str(i + 1)].Set(
                self.params["b_" + str(i + 1)]().Compile() - self.model.ale.W,
                definedon=self.compartment.domain,
            )
            self.params["gfu_rhs_" + str(i + 1)].Set(
                self.params["rhs_" + str(i + 1)]().Compile(), definedon=self.compartment.domain
            )
            self.params["gfu_u_bnd_" + str(i + 1)].Set(
                self.params["u_bnd_" + str(i + 1)]().Compile(), definedon=self.compartment.boundary
            )
            self.params["gfu_gradu_bnd_" + str(i + 1)].Set(
                self.params["gradu_bnd_" + str(i + 1)]().Compile(),
                definedon=self.compartment.boundary,
            )
            self.params["gfu_tot_flux_bnd_" + str(i + 1)].Set(
                self.params["tot_flux_bnd_" + str(i + 1)]().Compile(),
                definedon=self.compartment.boundary,
            )

        self.A.Assemble()
        self.invA.Update()

        self.F.Assemble()
        self.gfu.vec.data = self.invA * self.F.vec

        self.model.time.helper.t.Set(self.model.time.t.Get())

        for i in range(self.sys_dim):
            if (
                self.params["bounds_" + str(i + 1)]
                and not self.params["mass_preserving_" + str(i + 1)]
            ):
                if hasattr(self.model.time, "dt"):
                    dt = self.model.dt.Get()
                else:
                    logger.error(
                        "A time-dependent simulation is needed to impose conservative mass"
                    )

                self.Amp.Assemble()
                rows, cols, vals = self.Amp.mat.COO()
                weights = sp.csr_matrix((vals, (rows, cols))).diagonal()
                gfu_vec = self.sol[i].vec.Copy().FV().NumPy()

                gfu_new = MandBP(
                    gfu_vec,
                    weights=weights,
                    BP=self.params["bounds_" + str(i + 1)],
                    MP=True,
                    mass0=np.sum(weights * gfu_vec),
                    dt=dt,
                )
                self.sol[i].vec.data = gfu_new

            elif self.params["mass_preserving_" + str(i + 1)]:
                if hasattr(self.model.time, "dt"):
                    dt = self.model.dt.Get()
                else:
                    logger.error(
                        "A time-dependent simulation is needed to impose conservative mass"
                    )

                self.Amp.Assemble()
                rows, cols, vals = self.Amp.mat.COO()
                weights = sp.csr_matrix((vals, (rows, cols))).diagonal()
                gfu_vec = self.sol[i].vec.Copy().FV().NumPy()

                if self.params["bounds_" + str(i + 1)]:
                    BP = self.params["bounds_" + str(i + 1)]
                else:
                    BP = [-np.inf, np.inf]

                gfu_new = MandBP(
                    gfu_vec,
                    weights=weights,
                    BP=BP,
                    MP=self.params["mass_preserving_" + str(i + 1)],
                    mass0=self.mass0[i],
                    dt=dt,
                )

                self.sol[i].vec.data = gfu_new

    def PostProcess(self):

        pass

    def _base_env(self):
        # whitelist of functions (extend as needed)
        return {
            "sin": sin,
            "cos": cos,
            "exp": exp,
            "log": log,
            "sqrt": sqrt,
            "IfPos": IfPos,
            "x": x,
            "y": y,
            "z": z,
        }

    def add_nonlinearity(self, target, expression, map=None):
        if map is None:
            map = {}

        nonlin = {}
        nonlin["expr"] = expression
        nonlin["map"] = map
        nonlin["target"] = target

        self.nonlinearities.append(nonlin)
