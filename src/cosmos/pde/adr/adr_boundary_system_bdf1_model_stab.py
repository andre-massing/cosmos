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


class ADRBoundarySystemBDF1StabModel(BasePDEModel):
    """BDF1 advection–diffusion–reaction system on a surface compartment with gradient-jump stabilisation.

    Augments the standard SIPG bilinear form with gradient-jump penalty terms
    scaled by the local Péclet number, improving robustness in
    advection-dominated regimes. Uses a mixed H1 × NormalFacetSurface finite
    element space; the stabilisation flux unknowns are eliminated locally. All
    other features match the unstabilised variant.
    """

    is_bnd = True
    is_vol = False

    def __init__(
        self,
        name: str = "ADRBoundarySystemBDF1StabModel",
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
                "Parameter dim is needed for initialization of ADRBoundarySystemBDF1StabModel"
            )

        self.params["Neu_bnd"] = ""
        self.params["Dir_bnd"] = ""
        self.params["subdivision"] = 0
        self.params["fes_order"] = 1
        self.params["conservative"] = False

        self.V = H1(model.parentmesh, order=self.params["fes_order"], definedon=compartment.domain)
        if self.model.dim == 2:
            dV = H1(
                self.model.parentmesh, order=self.params["fes_order"], definedon=compartment.domain
            )
        else:
            dV = NormalFacetSurface(
                self.model.parentmesh,
                order=self.params["fes_order"] - 1,
                definedon=compartment.domain,
            )
        self.fes = self.V * dV
        for i in range(self.sys_dim - 1):
            self.fes = self.fes * self.V * dV

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

            self.params["gfu_b_" + str(i + 1)] = GridFunction(V2)
            self.params["gfu_d_" + str(i + 1)] = GridFunction(self.V)
            self.params["gfu_c_" + str(i + 1)] = GridFunction(self.V)
            self.params["gfu_rhs_" + str(i + 1)] = GridFunction(self.V)
            self.params["gfu_gradu_bnd_" + str(i + 1)] = GridFunction(V2)
            self.params["gfu_u_bnd_" + str(i + 1)] = GridFunction(self.V)

        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)

        self.sol = [self.gfu.components[2 * i] for i in range(self.sys_dim)]
        if self.model.dim == 2:
            self.vtk_gfu = [
                GridFunction(H1(self.model.parentmesh, order=self.params["fes_order"]))
                for i in range(self.sys_dim)
            ]
        else:
            self.vtk_gfu = self.sol

        ir_segm = IntegrationRule(points=[(0, 0), (1, 0)], weights=[1 / 2, 1 / 2])
        ir_trig = IntegrationRule(points=[(0, 0), (1, 0), (0, 1)], weights=[1 / 6, 1 / 6, 1 / 6])
        self.mass0 = np.zeros(self.sys_dim)
        ds_lumped = ds(intrules={SEGM: ir_segm, TRIG: ir_trig}, deformation=self.model.ale.Y)
        space = H1(model.parentmesh, order=self.params["fes_order"], definedon=compartment.domain)
        self.Amp = BilinearForm(space, symmetric=True)
        u, v = space.TnT()
        self.Amp += u * v * ds_lumped
        self.Amp.Assemble()
        rows, cols, vals = self.Amp.mat.COO()
        self.weights0 = sp.csr_matrix((vals, (rows, cols))).diagonal()

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

        if self.model.dim == 2:
            for i, gfu in enumerate(self.vtk_gfu):
                gfu.Set(self.sol[i], definedon=self.compartment.domain)

        trial, test = self.fes.TnT()
        self.A = BilinearForm(self.fes)
        self.F = LinearForm(self.fes)
        self.ns = specialcf.normal(self.model.dim)
        h = self.cfg.h
        alpha = 5 * self.params["fes_order"] * (self.params["fes_order"] + 1)
        tE = specialcf.tangential(self.model.dim)
        if self.model.dim == 2:
            facet_space = self.V
            nE = specialcf.tangential(self.model.dim)
        else:
            facet_space = FacetSurface(self.model.parentmesh, order=0)
            nE = Cross(self.ns, tE)

        for i in range(self.sys_dim):
            self.A += (
                self.params["gfu_c_" + str(i + 1)]
                * trial[2 * i]
                * test[2 * i]
                * ds(deformation=self.Yhalf)
            )
            self.A += (
                self.params["gfu_d_" + str(i + 1)]
                * grad(trial[2 * i]).Trace()
                * grad(test[2 * i]).Trace()
                * ds(deformation=self.Yhalf)
            )

            if self.params["Dir_bnd"]:
                dir_bnd_gfu = GridFunction(facet_space)
                dir_bnd_gfu.Set(
                    1, definedon=self.model.parentmesh.BBoundaries(self.params["Dir_bnd"])
                )
                self.A += (
                    -dir_bnd_gfu
                    * self.params["gfu_d_" + str(i + 1)]
                    * InnerProduct(nE, grad(trial[2 * i]).Trace())
                    * test[2 * i]
                    * ds(element_boundary=True, deformation=self.Yhalf)
                    - dir_bnd_gfu
                    * self.params["gfu_d_" + str(i + 1)]
                    * InnerProduct(nE, grad(test[2 * i]).Trace())
                    * trial[2 * i]
                    * ds(element_boundary=True, deformation=self.Yhalf)
                    + dir_bnd_gfu
                    * self.params["gfu_d_" + str(i + 1)]
                    * alpha
                    / h
                    * trial[2 * i]
                    * test[2 * i]
                    * ds(element_boundary=True, deformation=self.Yhalf)
                )

            self.A += (
                -(self.params["gfu_b_" + str(i + 1)])
                * grad(test[2 * i]).Trace()
                * trial[2 * i]
                * ds(deformation=self.Yhalf)
            )
            bnd_gfu = GridFunction(facet_space)
            bnd_gfu.Set(
                1,
                definedon=self.model.parentmesh.BBoundaries(
                    self.params["Dir_bnd"] + "|" + self.params["Neu_bnd"]
                ),
            )
            self.A += (
                bnd_gfu
                * IfPos(
                    (self.params["gfu_b_" + str(i + 1)]) * nE,
                    (self.params["gfu_b_" + str(i + 1)]) * nE * trial[2 * i],
                    CF(0),
                )
                * test[2 * i]
                * ds(element_boundary=True, deformation=self.Yhalf)
            )

            if self.model.dim == 2:
                tEc = CF((-self.ns[1], self.ns[0]))
                jump_dudn = (trial[2 * i].Trace().Deriv() - trial[2 * i + 1] * tEc) * nE
                jump_dvdn = (test[2 * i].Trace().Deriv() - test[2 * i + 1] * tEc) * nE
            elif self.model.dim == 3:
                jump_dudn = (trial[2 * i].Trace().Deriv() - trial[2 * i + 1].Trace()) * nE
                jump_dvdn = (test[2 * i].Trace().Deriv() - test[2 * i + 1].Trace()) * nE
            stab = Norm((self.params["gfu_b_" + str(i + 1)])) * h**2
            self.A += IfPos(
                stab,
                stab * InnerProduct(jump_dudn, jump_dvdn),
                InnerProduct(trial[2 * i + 1].Trace(), test[2 * i + 1].Trace()),
            ) * ds(element_boundary=True, deformation=self.Yhalf)
            self.A += (
                -1
                * bnd_gfu
                * IfPos(
                    stab,
                    stab * InnerProduct(jump_dudn, jump_dvdn),
                    InnerProduct(trial[2 * i + 1].Trace(), test[2 * i + 1].Trace()),
                )
                * ds(element_boundary=True, deformation=self.Yhalf)
            )
            self.A += (
                bnd_gfu
                * InnerProduct(trial[2 * i + 1].Trace(), test[2 * i + 1].Trace())
                * ds(element_boundary=True, deformation=self.Yhalf)
            )

            self.F += (
                self.params["gfu_rhs_" + str(i + 1)] * test[2 * i] * ds(deformation=self.Yhalf)
            )

            if self.params["Dir_bnd"]:
                self.F += dir_bnd_gfu * self.params[
                    "gfu_d_" + str(i + 1)
                ] * alpha / h * self.params["gfu_u_bnd_" + str(i + 1)] * test[2 * i] * ds(
                    element_boundary=True, deformation=self.Yhalf
                ) - dir_bnd_gfu * self.params["gfu_d_" + str(i + 1)] * InnerProduct(
                    nE, grad(test[2 * i]).Trace()
                ) * self.params["gfu_u_bnd_" + str(i + 1)] * ds(
                    element_boundary=True, deformation=self.Yhalf
                )
            if self.params["Neu_bnd"]:
                neu_bnd_gfu = GridFunction(facet_space)
                neu_bnd_gfu.Set(
                    1, definedon=self.model.parentmesh.BBoundaries(self.params["Neu_bnd"])
                )
                self.F += (
                    neu_bnd_gfu
                    * self.params["gfu_d_" + str(i + 1)]
                    * self.params["gfu_gradu_bnd_" + str(i + 1)]
                    * nE
                    * test[2 * i]
                    * ds(element_boundary=True, deformation=self.Yhalf)
                )

            self.F += (
                -bnd_gfu
                * IfPos(
                    (self.params["gfu_b_" + str(i + 1)]) * nE,
                    CF(0),
                    (self.params["gfu_b_" + str(i + 1)])
                    * nE
                    * self.params["gfu_u_bnd_" + str(i + 1)],
                )
                * test[2 * i]
                * ds(element_boundary=True, deformation=self.Yhalf)
            )

            self.A += (
                1 / self.model.dt * trial[2 * i] * test[2 * i] * ds(deformation=self.model.ale.Y)
            )

            self.F += (
                1
                / self.model.dt
                * self.gfu_old.components[2 * i]
                * test[2 * i]
                * ds(deformation=self.model.ale.Yo)
            )
            
        for nonlin in self.nonlinearities:
            env = {"__builtins__": {}}
            env.update(self._base_env())
            for j in range(self.sys_dim):
                env.update({"u" + str(j + 1): self.sol[j], "v" + str(j + 1): test[2 * j]})
            env.update(nonlin["map"])

            self.F += (
                -1
                * eval(nonlin["expr"], env)
                * test[2 * (nonlin["target"] - 1)]
                * ds(deformation=self.Yhalf)
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
                self.params["u_bnd_" + str(i + 1)]().Compile(), definedon=self.compartment.domain
            )
            self.params["gfu_gradu_bnd_" + str(i + 1)].Set(
                self.params["gradu_bnd_" + str(i + 1)]().Compile(),
                definedon=self.compartment.domain,
            )

        self.A.Assemble()
        self.invA.Update()

        self.F.Assemble()
        self.gfu.vec.data = self.invA * self.F.vec

        self.model.time.helper.t.Set(self.model.time.t.Get())

        self.gfu.vec.data = self.invA * self.F.vec

        for i in range(self.sys_dim):
            if (
                self.params["bounds_" + str(i + 1)]
                and not self.params["mass_preserving_" + str(i + 1)]
            ):
                if hasattr(self.model.time, "dt"):
                    dt = self.model.dt.Get()
                else:
                    raise ValueError(
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
                    raise ValueError(
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

        if self.model.dim == 2:
            for i, gfu in enumerate(self.vtk_gfu):
                gfu.Set(self.sol[i], definedon=self.compartment.domain)

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
