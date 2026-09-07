"""Surface shape evolution -- the model that actually drives ALE motion in
most Cosmos simulations, unlike the ADR/distance models which only ever
read the mesh velocity. Its output ``V_h`` (normal velocity) is what a
script hands to ``ale.set_normal_velocity`` after calling
``CosmosModel.create_ale`` on the same compartment; this file has no
awareness of the ALE manager itself, the wiring is done entirely by the
calling script/application.

Two things worth knowing before changing ``alpha``/``beta``/``gamma``:

- Setting only ``alpha`` (pure Willmore/bending energy, the class default)
  does *not* shrink a sphere -- a sphere is already the global Willmore-
  energy minimiser, so it's a steady state of that flow on its own (see
  ``tests/pde_examples/test_curvature_flow_on_a_sphere.py``). Classical
  *mean-curvature* flow (shrink-to-a-point behaviour) comes from the
  ``gamma`` (area-elasticity) term instead, with ``alpha=beta=0``.
- The sign convention: both ``InnerProduct(V, phi)`` and
  ``-gamma*InnerProduct(kappa, phi)`` are added to the *bilinear* form
  (``self.A``, the left-hand side) in ``Initialize()``, not the linear form,
  so with ``alpha=beta=0`` the assembled equation is ``V - gamma*kappa =
  0``, i.e. ``V = +gamma*kappa`` -- ``gamma`` needs to be *positive* to
  shrink a shape whose mean curvature ``kappa`` is negative (this package's
  sign convention for a sphere, see ``GeometricalFlowStationaryModel``'s
  own ``Initialize()``). Getting this sign backwards produces the
  ill-posed *expanding* flow, not merely a slower one.
"""

import logging

logger = logging.getLogger(__name__)

from typing import Optional

import numpy as np

from ngsolve import *
from cosmos.pde.base import BasePDEModel
from cosmos.core.field import Field
from cosmos.core.compartment import CosmosCompartment
from cosmos.core.model import CosmosModel


class GeometricalFlowModel(BasePDEModel):
    """BDF1 Willmore / mean-curvature flow model on a surface compartment.

    Evolves a surface by coupling a normal-velocity field to the mean curvature
    through a linearised Willmore-flow energy with coefficients ``alpha``
    (bending), ``beta`` (surface tension), and ``gamma`` (area elasticity).
    Supports a prescribed spontaneous curvature, clamped and Navier boundary
    conditions, and optional area- and/or volume-preserving Lagrange-multiplier
    constraints solved by an inner fixed-point iteration.
    """

    is_bnd = True
    is_vol = False

    def __init__(
        self,
        name: str = "GeometricalFlowModel",
        model: Optional[CosmosModel] = None,
        compartment: Optional[CosmosCompartment] = None,
    ):

        super().__init__(name=name, model=model, compartment=compartment)

        self.cn_bboundary = compartment.clamped_bbnd + "|" + compartment.navier_bbnd
        self.navier_bbnd = compartment.navier_bbnd
        self.params["subdivision"] = 0
        self.params["sp_curv"] = CF(0)
        self.params["rhs"] = Field(CF(0))
        self.params["alpha"] = CF(1)
        self.params["beta"] = CF(0)
        self.params["gamma"] = CF(0)
        self.params["area_preserving"] = False
        self.params["volume_preserving"] = False
        self.params["kappa0"] = None

        self.vectorspace_bbnd = VectorH1(
            model.parentmesh,
            order=1,
            definedon=compartment.domain,
            dirichlet_bbnd=self.cn_bboundary,
        )
        self.vectorspace = VectorH1(model.parentmesh, order=1, definedon=compartment.domain)
        self.scalarspace = H1(model.parentmesh, order=1, definedon=compartment.domain)
        self.scalarspace_bbnd = H1(
            model.parentmesh,
            order=1,
            definedon=compartment.domain,
            dirichlet_bbnd=self.cn_bboundary,
        )
        self.scalarspace_navier_bbnd = H1(
            model.parentmesh, order=1, definedon=compartment.domain, dirichlet_bbnd=self.navier_bbnd
        )
        self.discscalarspace = SurfaceL2(model.parentmesh, order=0, definedon=compartment.domain)

        self.fes = self.scalarspace_bbnd * self.scalarspace_navier_bbnd
        self.gfu = GridFunction(self.fes)
        self.V_h, self.kappa_h = self.gfu.components

        dim = model.dim
        self.ns = specialcf.normal(dim)
        self.Ps = Id(dim) - OuterProduct(self.ns, self.ns)
        if model.dim == 2:
            self.identity = CF((x, y))
        else:
            self.identity = CF((x, y, z))

        self.pre_normal = GridFunction(self.vectorspace)
        self.normal = GridFunction(self.vectorspace)
        self.Amap_h = GridFunction(self.vectorspace_bbnd)
        self.kappa_h_old = GridFunction(self.scalarspace_navier_bbnd)
        self.W_h_old = GridFunction(self.discscalarspace)
        self.J_h_old = GridFunction(self.discscalarspace)

        self.sp_curv_gfu = GridFunction(self.scalarspace_bbnd)

        self.lam = GridFunction(NumberSpace(model.parentmesh, definedon=compartment.domain))
        self.mu = GridFunction(NumberSpace(model.parentmesh, definedon=compartment.domain))
        self.one = GridFunction(NumberSpace(model.parentmesh, definedon=compartment.domain))
        self.one.Set(1, definedon=compartment.domain)

        if self.model.dim == 2:
            self.vtk_gfu = [GridFunction(H1(self.model.parentmesh, order=1)) for i in range(2)]
        else:
            self.vtk_gfu = self.gfu.components
        self.vtk_names = [self.name + "_V", self.name + "_kappa"]

    def Initialize(self):

        if self.params["kappa0"]:
            self.kappa_h.Set(self.params["kappa0"], dual=True, definedon=self.compartment.domain)
        else:
            fes0 = self.vectorspace_bbnd * self.scalarspace_bbnd
            (dY0, kappa0), (nu0, zeta0) = fes0.TnT()
            gfu0 = GridFunction(fes0)
            dY0_h, kappa0_h = gfu0.components

            ir_segm = IntegrationRule(points=[(0, 0), (1, 0)], weights=[1 / 2, 1 / 2])
            ir_trig = IntegrationRule(
                points=[(0, 0), (1, 0), (0, 1)], weights=[1 / 6, 1 / 6, 1 / 6]
            )

            A0 = BilinearForm(fes0)
            A0 += InnerProduct(dY0 * self.ns, zeta0) * ds(
                intrules={SEGM: ir_segm, TRIG: ir_trig}, deformation=self.model.ale.Yo
            )
            A0 += InnerProduct(kappa0 * self.ns, nu0) * ds(
                intrules={SEGM: ir_segm, TRIG: ir_trig}, deformation=self.model.ale.Yo
            )
            A0 += InnerProduct(grad(dY0).Trace(), grad(nu0).Trace()) * ds(
                deformation=self.model.ale.Yo
            )
            A0.Assemble()
            invA0 = A0.mat.Inverse(freedofs=fes0.FreeDofs())

            F0 = LinearForm(fes0)
            F0 += -InnerProduct(self.Ps, grad(nu0).Trace()) * ds(deformation=self.model.ale.Yo)
            F0.Assemble()

            gfu0.vec.data = invA0 * F0.vec
            self.kappa_h.vec.data = kappa0_h.vec.data

        if self.model.dim == 2:
            if self.params["printing"]:
                for i, gfu in enumerate(self.vtk_gfu):
                    gfu.Set(self.gfu.components[i], definedon=self.compartment.domain)

        alpha = self.params["alpha"]
        beta = self.params["beta"]
        gamma = self.params["gamma"]
        self.gfu_rhs = GridFunction(self.scalarspace)

        (V, kappa), (phi, xsi) = self.fes.TnT()

        self.A = BilinearForm(self.fes)
        self.A += InnerProduct(V, phi) * ds(deformation=self.model.ale.Yo)
        self.A += (
            -alpha
            * InnerProduct(grad(kappa).Trace(), grad(phi).Trace())
            * ds(deformation=self.model.ale.Yo)
        )
        self.A += (
            alpha * InnerProduct(self.W_h_old * kappa, phi) * ds(deformation=self.model.ale.Yo)
        )
        self.A += (
            -alpha
            * 0.5
            * InnerProduct(
                (self.kappa_h_old - self.params["sp_curv"]) * self.kappa_h_old * kappa, phi
            )
            * ds(deformation=self.model.ale.Yo)
        )
        self.A += (
            -beta
            * InnerProduct(grad(kappa).Trace(), grad(phi).Trace())
            * ds(deformation=self.model.ale.Yo)
        )
        self.A += -gamma * InnerProduct(kappa, phi) * ds(deformation=self.model.ale.Yo)

        self.A += InnerProduct(kappa / self.model.dt, xsi) * ds(deformation=self.model.ale.Yo)
        self.A += (
            -0.5
            * (
                InnerProduct(self.model.ale.Wo, grad(kappa).Trace() * xsi)
                - InnerProduct(self.model.ale.Wo, grad(xsi).Trace() * kappa)
            )
            * ds(deformation=self.model.ale.Yo)
        )
        self.A += InnerProduct(grad(V).Trace(), grad(xsi).Trace()) * ds(
            deformation=self.model.ale.Yo
        )
        self.A += -InnerProduct(self.W_h_old * V, xsi) * ds(deformation=self.model.ale.Yo)
        self.A += (
            0.5
            * InnerProduct(V, (self.kappa_h_old - self.params["sp_curv"]) * self.kappa_h_old * xsi)
            * ds(deformation=self.model.ale.Yo)
        )

        self.A += -1 * InnerProduct(self.lam * kappa, phi) * ds(deformation=self.model.ale.Yo)

        self.A.Assemble()
        self.invA = self.A.mat.Inverse(freedofs=self.fes.FreeDofs())

        self.F = LinearForm(self.fes)

        self.F += InnerProduct(self.gfu_rhs, phi) * ds(deformation=self.model.ale.Yo)

        self.F += (
            alpha
            * InnerProduct(self.W_h_old * self.params["sp_curv"], phi)
            * ds(deformation=self.model.ale.Yo)
        )
        self.F += (
            -alpha
            * 0.5
            * InnerProduct(
                (self.kappa_h_old - self.params["sp_curv"])
                * self.kappa_h_old
                * self.params["sp_curv"],
                phi,
            )
            * ds(deformation=self.model.ale.Yo)
        )

        self.F += InnerProduct(self.params["sp_curv"] / self.model.dt, xsi) * ds(
            deformation=self.model.ale.Yo
        )
        self.F += InnerProduct(
            (self.kappa_h_old - self.params["sp_curv"]) / self.model.dt * sqrt(self.J_h_old), xsi
        ) * ds(deformation=self.model.ale.Yo)
        self.F += (
            -0.5
            * (-InnerProduct(self.model.ale.Wo, grad(xsi).Trace() * self.params["sp_curv"]))
            * ds(deformation=self.model.ale.Yo)
        )

        self.F += InnerProduct(self.mu, phi) * ds(deformation=self.model.ale.Yo)

        logger.info(
            f"[Cosmos] PDE '{self.name}' ({type(self).__name__}) initialized "
            f"on compartment '{self.compartment.name}'"
        )

    def PreProcess(self):

        self.kappa_h_old.vec.data = self.kappa_h.vec.data
        self.pre_normal.Set(self.ns, dual=True, definedon=self.compartment.domain)
        self.normal.Set(Normalize(self.pre_normal), dual=True, definedon=self.compartment.domain)
        self.W_h_old.Set(Norm(grad(self.normal).Trace()) ** 2, definedon=self.compartment.domain)

        self.Amap_h.Set(
            self.identity - self.model.dt * self.model.ale.Wo,
            dual=True,
            definedon=self.compartment.domain,
        )
        self.J_h_old.Set(
            sqrt(
                Det(
                    Grad(self.Amap_h).Trace().trans * Grad(self.Amap_h).Trace()
                    + OuterProduct(self.ns, self.ns)
                )
            ),
            definedon=self.compartment.domain,
        )

    def Solve(self):

        self.gfu_rhs.Set(self.params["rhs"](), definedon=self.compartment.domain)

        if self.params["area_preserving"] or self.params["volume_preserving"]:
            self.A.Assemble()
            self.invA.Update()

            iter = 0
            lam_old = 0
            lam_new = 0
            mu_old = 0
            mu_new = 0
            tol = 1e-6

            max_iter = 10

            while iter < max_iter:
                lam_old = lam_new
                mu_old = mu_new
                self.lam.vec[:] = lam_old
                self.mu.vec[:] = mu_old

                iter += 1

                self.kappa_h.Set(
                    self.params["sp_curv"],
                    definedon=self.model.parentmesh.BBoundaries(self.navier_bbnd),
                )
                self.V_h.vec.data[:] = 0

                if self.params["area_preserving"] and not self.params["volume_preserving"]:
                    self.A.Assemble()
                    self.invA.Update()
                    self.F.Assemble()
                    res = self.A.mat * self.gfu.vec
                    self.gfu.vec.data += self.invA * (self.F.vec - res)

                    lam_new = Integrate(
                        (-self.V_h + self.lam * self.kappa_h) * self.kappa_h,
                        mesh=self.model.parentmesh,
                        VOL_or_BND=BND,
                    ) / Integrate(self.kappa_h**2, mesh=self.model.parentmesh, VOL_or_BND=BND)

                elif self.params["volume_preserving"] and not self.params["area_preserving"]:
                    self.F.Assemble()
                    res = self.A.mat * self.gfu.vec
                    self.gfu.vec.data += self.invA * (self.F.vec - res)

                    mu_new = Integrate(
                        -self.V_h + self.mu, mesh=self.model.parentmesh, VOL_or_BND=BND
                    ) / Integrate(self.one, mesh=self.model.parentmesh, VOL_or_BND=BND)

                elif self.params["volume_preserving"] and self.params["area_preserving"]:
                    self.A.Assemble()
                    self.invA.Update()
                    self.F.Assemble()
                    res = self.A.mat * self.gfu.vec
                    self.gfu.vec.data += self.invA * (self.F.vec - res)

                    M = np.zeros((2, 2))
                    c = np.zeros(2)
                    M[0, 0] = Integrate(self.kappa_h**2, mesh=self.model.parentmesh, VOL_or_BND=BND)
                    M[0, 1] = Integrate(self.kappa_h, mesh=self.model.parentmesh, VOL_or_BND=BND)
                    M[1, 0] = Integrate(self.kappa_h, mesh=self.model.parentmesh, VOL_or_BND=BND)
                    M[1, 1] = Integrate(self.one, mesh=self.model.parentmesh, VOL_or_BND=BND)
                    c[0] = Integrate(
                        (-self.V_h + self.lam * self.kappa_h + self.mu) * self.kappa_h,
                        mesh=self.model.parentmesh,
                        VOL_or_BND=BND,
                    )
                    c[1] = Integrate(
                        -self.V_h + self.lam * self.kappa_h + self.mu,
                        mesh=self.model.parentmesh,
                        VOL_or_BND=BND,
                    )
                    x = np.linalg.solve(M, c)
                    lam_new = x[0]
                    mu_new = x[1]

                err_lam = abs(lam_new - lam_old)
                err_mu = abs(mu_new - mu_old)
                if err_lam < tol and err_mu < tol:
                    # print('Converged in ', iter, ' iterations')
                    break

            if iter == max_iter:
                raise RuntimeError("Number of iterations for internal solver exceeded")

        else:
            self.kappa_h.Set(
                self.params["sp_curv"],
                definedon=self.model.parentmesh.BBoundaries(self.navier_bbnd),
            )
            self.V_h.vec.data[:] = 0

            self.A.Assemble()
            self.invA.Update()
            self.F.Assemble()

            res = self.A.mat * self.gfu.vec
            self.gfu.vec.data += self.invA * (self.F.vec - res)

    def PostProcess(self):

        if self.model.dim == 2:
            if self.params["printing"]:
                for i, gfu in enumerate(self.vtk_gfu):
                    gfu.Set(self.gfu.components[i], definedon=self.compartment.domain)
