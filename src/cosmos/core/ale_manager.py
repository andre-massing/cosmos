import logging

logger = logging.getLogger(__name__)

import time
from ngsolve import *
from cosmos.core.field import Field
from cosmos.core.compartment import CosmosCompartment
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cosmos.core.model import CosmosModel


class CosmosALEManager:
    """Manages ALE (Arbitrary Lagrangian–Eulerian) mesh motion for a CosmosModel.

    Maintains displacement, velocity, and reference-position GridFunctions for
    the whole mesh. At each time step it collects per-compartment displacements
    from registered boundary and volume ALE fields, optionally extends boundary
    displacements to the bulk via a Laplace or linear-elasticity solve, and
    updates the mesh deformation in NGSolve.
    """

    def __init__(self, model: "CosmosModel", kwargs):

        self.model = model
        self.params = kwargs
        if "volume_ALE" in kwargs:
            self.volume_ALE = kwargs["volume_ALE"]
        else:
            self.volume_ALE = "laplace"
        if "surface_ALE" in kwargs:
            self.surface_ALE = kwargs["surface_ALE"]
        else:
            self.surface_ALE = "mdr"
        self.redistribute = False
        self.ale_elapsed_time = None

        if self.model.is_bnd:
            self.domain = self.model.parentmesh.Boundaries(".*")
            self.fes = VectorH1(
                self.model.parentmesh, order=self.model.geo_order, definedon=self.domain
            )
            gfu = GridFunction(self.fes)
            self.model.parentmesh.SetDeformation(gfu)
        else:
            self.domain = self.model.parentmesh.Materials(".*")
            self.fes = VectorH1(
                self.model.parentmesh, order=self.model.geo_order, definedon=self.domain
            )
            gfu = GridFunction(self.fes)
            self.model.parentmesh.SetDeformation(gfu)

        self.dY = GridFunction(self.fes)
        self.Y = GridFunction(self.fes)
        self.Yo = GridFunction(self.fes)
        self.Xo = GridFunction(self.fes)
        self.X = GridFunction(self.fes)
        if self.model.dim == 2:
            self.Xo.Set(CF((x, y)), definedon=self.domain)
            self.X.Set(CF((x, y)), definedon=self.domain)
        elif self.model.dim == 3:
            self.Xo.Set(CF((x, y, z)), definedon=self.domain)
            self.X.Set(CF((x, y, z)), definedon=self.domain)
        self.W = GridFunction(self.fes)
        self.Wo = GridFunction(self.fes)
        self.gfu_bnd_ale = GridFunction(self.fes)
        self.gfu_bnd_mat = GridFunction(self.fes)
        self.gfu_vol_ale = GridFunction(self.fes)
        self.gfu_vol_mat = GridFunction(self.fes)
        self.dY_mat = GridFunction(self.fes)
        self.V = GridFunction(self.fes)
        self.Vo = GridFunction(self.fes)

        if "redistribute" in self.params.keys():
            if isinstance(self.params["redistribute"], bool):
                self.redistribute = self.params["redistribute"]
            else:
                raise ValueError(
                    f"Mesh redistribution for model {self.model.name} must be either True or False"
                )

        if self.model.is_vol:
            if self.volume_ALE == "laplace":
                self.V0 = VectorH1(
                    self.model.parentmesh,
                    order=self.model.geo_order,
                    dirichlet=self.model.parentmesh.Boundaries(".*"),
                )
                u, v = self.V0.TnT()
                self.A = BilinearForm(self.V0, symmetric=True)
                self.A += InnerProduct(Grad(u), Grad(v)) * dx(deformation=self.Yo)
                self.A.Assemble()
                self.invA = self.A.mat.Inverse(freedofs=self.V0.FreeDofs())

            elif self.volume_ALE == "linel":
                self.V0 = VectorH1(
                    self.model.parentmesh,
                    order=self.model.geo_order,
                    dirichlet=self.model.parentmesh.Boundaries(".*"),
                )
                u, v = self.V0.TnT()
                self.A = BilinearForm(self.V0)
                h = specialcf.mesh_size
                E, nu = 1 / h**self.model.dim, 0.49
                mu = E / 2 / (1 + nu)
                lam = E * nu / ((1 + nu) * (1 - 2 * nu))

                def Stress(strain):
                    return 2 * mu * strain + lam * Trace(strain) * Id(self.model.dim)

                self.A += InnerProduct(Stress(Sym(Grad(u))), Sym(Grad(v))) * dx(deformation=self.Yo)
                self.A.Assemble()
                self.invA = self.A.mat.Inverse(freedofs=self.V0.FreeDofs())

            else:
                raise ValueError("ALE extension to volume type not known")

        self.vtk_gfu = [self.dY, self.Y, self.X, self.W, self.V]
        self.vtk_names = ["ale_dY", "ale_Y", "ale_X", "ale_W", "ale_V"]

    def initialize(self):

        self.bnd_ales = {}
        self.bnd_mats = {}
        self.vol_ales = {}
        self.vol_mats = {}
        for ale in self.model.ales:
            if isinstance(ale, CosmosBndALEField):
                self.bnd_ales[ale.compartment.domain_id] = ale.ale_displ
                self.bnd_mats[ale.compartment.domain_id] = ale.mat_displ
            elif isinstance(ale, CosmosVolALEField):
                self.vol_ales[ale.compartment.domain_id] = ale.ale_displ
                self.vol_mats[ale.compartment.domain_id] = ale.mat_displ

        self.bnd_ale_cf = self.model.parentmesh.BoundaryCF(
            self.bnd_ales, default=CF((0,) * self.model.dim)
        )
        self.bnd_mat_cf = self.model.parentmesh.BoundaryCF(
            self.bnd_mats, default=CF((0,) * self.model.dim)
        )
        self.vol_ale_cf = self.model.parentmesh.MaterialCF(
            self.vol_ales, default=CF((0,) * self.model.dim)
        )
        self.vol_mat_cf = self.model.parentmesh.MaterialCF(
            self.vol_mats, default=CF((0,) * self.model.dim)
        )

    def solve_ale(self):

        start = time.time()

        self.dY.vec.data[:] = 0
        self.dY_mat.vec.data[:] = 0

        for ale in self.model.ales:
            ale.update(self.redistribute)

        if self.bnd_ales:
            if self.model.is_vol:
                self.A.Assemble()
                self.invA.Update()

            self.gfu_bnd_ale.Set(self.bnd_ale_cf, definedon=self.model.parentmesh.Boundaries(".*"))
            if self.model.is_vol:
                self._extend_displacement_to_bulk(self.gfu_bnd_ale)
            self.dY.vec.data += self.gfu_bnd_ale.vec.data

            self.gfu_bnd_mat.Set(self.bnd_mat_cf, definedon=self.model.parentmesh.Boundaries(".*"))
            if self.model.is_vol:
                self._extend_displacement_to_bulk(self.gfu_bnd_mat)
            self.dY_mat.vec.data += self.gfu_bnd_mat.vec.data

            # ns = specialcf.normal(self.model.dim)
            # Qs = OuterProduct(ns, ns)
            # Ps = Id(self.model.dim) - OuterProduct(ns, ns)
            # self.gfu_bnd_mat.Set(self.bnd_mat_cf, definedon = self.model.parentmesh.Boundaries('.*'))
            # self.gfu_bnd_ale.Set(self.bnd_ale_cf, definedon = self.model.parentmesh.Boundaries('.*'))
            # if self.model.is_vol:
            #     self._extend_displacement_to_bulk(self.gfu_bnd_ale)
            # self.dY.vec.data += self.gfu_bnd_ale.vec.data

            # if self.model.is_vol:
            #     self._extend_displacement_to_bulk(self.gfu_bnd_mat)
            # self.dY_mat.vec.data += self.gfu_bnd_mat.vec.data

        if self.vol_ales:
            self.gfu_vol_ale.Set(self.vol_ale_cf)
            self.gfu_vol_mat.Set(self.vol_mat_cf)

            self.dY.vec.data += self.gfu_vol_ale.vec.data
            self.dY_mat.vec.data += self.gfu_vol_mat.vec.data

        if self.bnd_ales or self.vol_ales:
            self.Y.vec.data = self.Yo.vec.data + self.dY.vec.data
            self.X.vec.data = self.Xo.vec.data + self.dY.vec.data
            self.W.Set(self.dY / self.model.dt, definedon=self.domain)
            self.V.Set(self.dY_mat / self.model.dt, definedon=self.domain)

        stop = time.time()

        self.ale_elapsed_time = stop - start

    def finalize(self):

        if self.bnd_ales or self.vol_ales:
            self.model.parentmesh.deformation.vec.data = self.Y.vec.data
            self.Yo.vec.data = self.Y.vec.data
            self.Xo.vec.data = self.X.vec.data
            self.Wo.vec.data = self.W.vec.data
            self.Vo.vec.data = self.V.vec.data

    def _extend_displacement_to_bulk(self, gfu):

        vec = -1 * self.A.mat * gfu.vec
        gfu.vec.data += self.invA * vec

    def reset(self):

        self.Y.vec.data = self.Yo.vec.data
        self.X.vec.data = self.Xo.vec.data
        self.W.vec.data = self.Wo.vec.data
        self.V.vec.data = self.Vo.vec.data


class CosmosBndALEField:
    """ALE displacement field defined on a boundary (surface) compartment.

    Prescribes the mesh displacement from a user-supplied normal velocity
    (and optionally a tangential velocity) on the associated surface. Several
    surface redistribution strategies are available: ``'mdr'`` (mesh-dependent
    redistribution), ``'gnz'``, ``'ms'``/``'ms0'`` (membrane-style), and
    ``'duanli'``.
    """

    def __init__(self, name: str, model: "CosmosModel", compartment: CosmosCompartment):

        self.name = name
        self.model = model
        self.compartment = compartment
        self.tangential_velocity = None
        self.normal_velocity = None
        self.domain_velocity = None

        self.ns = specialcf.normal(self.model.dim)
        self.Ps = Id(self.model.dim) - OuterProduct(self.ns, self.ns)
        self.Qs = OuterProduct(self.ns, self.ns)

        self.fes = VectorH1(
            self.model.parentmesh,
            order=self.model.geo_order,
            definedon=self.compartment.domain,
            dirichlet_bbnd=self.compartment.boundary,
        )
        S = H1(
            self.model.parentmesh,
            order=self.model.geo_order,
            definedon=self.compartment.domain,
            dirichlet_bbnd=self.compartment.boundary,
        )

        self.ale_displ = GridFunction(self.fes)
        self.mat_displ = GridFunction(self.fes)
        self.ale_vel = GridFunction(self.fes)
        self.mat_vel = GridFunction(self.fes)

        self.gfu_norm_vel = GridFunction(S)

        fes_pp = self.fes * S
        (dX_pp, kappa_pp), (nu_pp, zeta_pp) = fes_pp.TnT()
        self.gfu_pp = GridFunction(fes_pp)

        ir_segm = IntegrationRule(points=[(0, 0), (1, 0)], weights=[1 / 2, 1 / 2])
        ir_trig = IntegrationRule(points=[(0, 0), (1, 0), (0, 1)], weights=[1 / 6, 1 / 6, 1 / 6])

        if self.model.ale.surface_ALE == "duanli":
            gfu0 = GridFunction(self.model.ale.Yo.space)

            self.A_pp = BilinearForm(fes_pp, symmetric=True)
            self.A_pp += InnerProduct(dX_pp * self.ns, zeta_pp) * ds(
                intrules={SEGM: ir_segm, TRIG: ir_trig}, deformation=self.model.ale.Yo
            )
            self.A_pp += InnerProduct(kappa_pp * self.ns, nu_pp) * ds(
                intrules={SEGM: ir_segm, TRIG: ir_trig}, deformation=self.model.ale.Yo
            )
            self.A_pp += InnerProduct(Grad(dX_pp).Trace(), Grad(nu_pp).Trace()) * ds(
                deformation=gfu0
            )
            self.A_pp.Assemble()
            self.invA_pp = self.A_pp.mat.Inverse(freedofs=fes_pp.FreeDofs())

            self.F_pp = LinearForm(fes_pp)
            self.F_pp += InnerProduct(self.gfu_norm_vel * self.model.dt, zeta_pp) * ds(
                intrules={SEGM: ir_segm, TRIG: ir_trig}, deformation=self.model.ale.Yo
            )
            self.F_pp += (
                -1
                * InnerProduct(Grad(self.model.ale.Yo).Trace(), Grad(nu_pp).Trace())
                * ds(deformation=gfu0)
            )
            self.F_pp += -1 * InnerProduct(self.Ps, Grad(nu_pp).Trace()) * ds(deformation=gfu0)

        elif self.model.ale.surface_ALE == "mdr":
            self.A_pp = BilinearForm(fes_pp, symmetric=True)
            self.A_pp += InnerProduct(dX_pp * self.ns, zeta_pp) * ds(
                intrules={SEGM: ir_segm, TRIG: ir_trig}, deformation=self.model.ale.Yo
            )
            self.A_pp += InnerProduct(kappa_pp * self.ns, nu_pp) * ds(
                intrules={SEGM: ir_segm, TRIG: ir_trig}, deformation=self.model.ale.Yo
            )
            self.A_pp += InnerProduct(
                1 / self.model.dt * Grad(dX_pp).Trace(), Grad(nu_pp).Trace()
            ) * ds(deformation=self.model.ale.Yo)
            self.A_pp.Assemble()
            self.invA_pp = self.A_pp.mat.Inverse(freedofs=fes_pp.FreeDofs())

            self.F_pp = LinearForm(fes_pp)
            self.F_pp += InnerProduct(self.gfu_norm_vel * self.model.dt, zeta_pp) * ds(
                intrules={SEGM: ir_segm, TRIG: ir_trig}, deformation=self.model.ale.Yo
            )

        elif self.model.ale.surface_ALE == "gnz":
            self.A_pp = BilinearForm(fes_pp)
            self.A_pp += InnerProduct(dX_pp * self.ns, zeta_pp) * ds(
                intrules={SEGM: ir_segm, TRIG: ir_trig}, deformation=self.model.ale.Yo
            )
            self.A_pp += InnerProduct(kappa_pp * self.ns, nu_pp) * ds(
                intrules={SEGM: ir_segm, TRIG: ir_trig}, deformation=self.model.ale.Yo
            )
            self.A_pp += InnerProduct(Grad(dX_pp).Trace(), Grad(nu_pp).Trace()) * ds(
                deformation=self.model.ale.Yo
            )
            self.A_pp.Assemble()
            self.invA_pp = self.A_pp.mat.Inverse(freedofs=fes_pp.FreeDofs())

            self.F_pp = LinearForm(fes_pp)
            self.F_pp += InnerProduct(self.gfu_norm_vel * self.model.dt, zeta_pp) * ds(
                intrules={SEGM: ir_segm, TRIG: ir_trig}, deformation=self.model.ale.Yo
            )
            self.F_pp += (
                -1 * InnerProduct(self.Ps, Grad(nu_pp).Trace()) * ds(deformation=self.model.ale.Yo)
            )

        else:
            raise ValueError("Surface ALE distribution not known")

    def set_normal_velocity(self, coef):
        self.normal_velocity = Field(coef)

    def set_tangential_velocity(self, coef):
        self.tangential_velocity = Field(coef)

    def set_domain_velocity(self, coef):
        self.domain_velocity = Field(coef)

    def update(self, redistribute: bool):

        if self.domain_velocity is not None:
            self.ale_displ.Set(
                self.domain_velocity() * self.model.time.dt, definedon=self.compartment.domain
            )
            self.mat_displ.vec.data = self.ale_displ.vec.data
        else:
            if self.normal_velocity is None:
                raise ValueError(f"Normal velocity for ALE {self.name} must be set")
            if self.tangential_velocity is None:
                raise ValueError(f"Tangential velocity for ALE {self.name} must be set")

            if redistribute:
                self.gfu_norm_vel.Set(self.normal_velocity(), definedon=self.compartment.domain)
                self.ale_displ.vec.data[:] = 0
                self.A_pp.Assemble()
                self.invA_pp.Update()
                self.F_pp.Assemble()

                self.gfu_pp.vec.data = self.invA_pp * self.F_pp.vec
                self.ale_displ.vec.data = self.gfu_pp.components[0].vec.data
            else:
                self.gfu_norm_vel.Set(self.normal_velocity(), definedon=self.compartment.domain)
                self.ale_displ.Set(
                    (self.gfu_norm_vel * self.ns + self.Ps * self.tangential_velocity())
                    * self.model.time.dt,
                    definedon=self.compartment.domain,
                )
            self.mat_displ.Set(
                (self.normal_velocity() * self.ns + self.Ps * self.tangential_velocity())
                * self.model.time.dt,
                definedon=self.compartment.domain,
            )


class CosmosVolALEField:
    """ALE displacement field defined on a volume compartment.

    Prescribes the mesh motion inside a volumetric region through a user-supplied
    domain velocity CoefficientFunction. The computed displacement is passed back
    to :class:`CosmosALEManager` each step.
    """

    def __init__(self, name: str, model: "CosmosModel", compartment: CosmosCompartment):

        self.name = name
        self.model = model
        self.compartment = compartment
        self.domain_velocity = None

        self.fes = VectorH1(
            self.model.parentmesh, order=self.model.geo_order, definedon=self.compartment.domain
        )

        self.ale_displ = GridFunction(self.fes)
        self.mat_displ = GridFunction(self.fes)
        self.ale_vel = GridFunction(self.fes)
        self.mat_vel = GridFunction(self.fes)

    def set_domain_velocity(self, coef):
        self.domain_velocity = Field(coef)

    def update(self, redistribute: bool):

        if self.domain_velocity is None:
            raise ValueError(f"Domain velocity for ALE {self.name} must be set")

        self.ale_displ.Set(
            self.domain_velocity() * self.model.time.dt, definedon=self.compartment.domain
        )
        self.mat_displ.vec.data = self.ale_displ.vec.data
