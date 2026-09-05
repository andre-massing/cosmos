"""Convergence study: Convergence tests for a diffusion-driven
    mean curvature flow of a sphere.
"""

import numpy as np
import pytest
from ngsolve import *
from cosmos import *
from ngsolve.webgui import Draw
from cosmos.utils.generate_meshes import generate_boundary_sphere
from cosmos.utils.tools import gradient
from cosmos.core.model import CosmosModel
from cosmos.pde import (
    GeometricalFlowModel,
    GeometricalFlowStationaryModel,
    ADRBoundarySystemBDF1Model,
    ADRBoundarySystemBDF1StabModel
)

pytestmark = pytest.mark.convergence
pytestmark = pytest.mark.slow

@pytest.mark.parametrize("redistribute", [False, True])
@pytest.mark.parametrize("adr_solver", [ADRBoundarySystemBDF1Model, ADRBoundarySystemBDF1StabModel])
@pytest.mark.parametrize("geom_flow_sovler", [GeometricalFlowModel, GeometricalFlowStationaryModel])
def test_convergence_coupling_adr_geom_flow(
    redistribute,
    adr_solver,
    geom_flow_sovler
    ):

    R0 = 1
    R1 = 2
    kappa = 0.5
    delta = 0.4
    Tend = 1

    def solve(mesh, dt):

        gfu = GridFunction(H1(mesh, order=1, definedon=mesh.Boundaries(".*")))

        t = Parameter(0.0)
        dt = Parameter(dt)

        Rt = R0 * R1 / (R1 * exp(-kappa * t) + R0 * (1 - exp(-kappa * t)))

        u0 = x * y
        u_ex = u0 * exp(-6 * t)
        v = Rt.Diff(t)
        n = Normalize(CF((x, y, z)))
        Vn = v * n
        P = Id(3) - OuterProduct(n, n)
        f = (
            u_ex.Diff(t)
            + Vn * gradient(u_ex, Id(3))
            + u_ex * Trace(gradient(Vn, P))
            - Trace(gradient(gradient(u_ex, P), P))
        )
        g = v + 2 / Rt - delta * u_ex

        model = CosmosModel(
            name="test_convergence_coupling_adr_geom_flow",
            parentmesh=mesh,
            t0=0,
            t1=Tend,
            dt=dt,
            t=t,
            coupling_type="implicit",
            redistribute=redistribute,
        )
        comp1 = model.create_compartment(name="compartment", boundary="default", bboundary="")
        mc = model.create_pde(
            name="mean_curvature", pde_model=geom_flow_sovler, compartment=comp1, ale_type=0
        )
        adr = model.create_pde(
            name="adr", pde_model=adr_solver, compartment=comp1, ale_type=1, dim=1
        )
        ale = model.create_ale(name="ale", compartment=comp1)

        ale.set_normal_velocity(mc.V_h)
        ale.set_tangential_velocity(CF((0, 0, 0)))

        mc.set_params(alpha=0, beta=0, gamma=1, rhs=lambda: delta * adr.sol[0] + g)

        adr.set_params(
            u0_1=u0,
            d_1=1,
            rhs_1=f,
            b_1=lambda: model.ale.V,
        )

        errs = []
        for _ in model():
            gfu.Set(sqrt(x**2 + y**2 + z**2) - Rt, dual=True, definedon=mesh.Boundaries(".*"))
            errs.append(np.max(np.abs(gfu.vec.FV().NumPy())))

        return np.max(errs)

    power_t = 1.5
    dt0 = 0.05
    dt_refs = 3
    dts = dt0 / (power_t ** (np.arange(dt_refs + 1)))

    power_h = 1.5
    dh0 = 0.1
    dh_refs = 3
    dhs = dh0 / (power_h ** (np.arange(dh_refs + 1)))

    ERRORS_dX = np.zeros((len(dts), len(dhs)))
    ERRORS_dX.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):
            mesh = generate_boundary_sphere(maxh=dh, R=R0)

            true_h = GridFunction(SurfaceL2(mesh, order=0))
            true_h.Set(get_config().h, definedon=mesh.Boundaries(".*"))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            err = solve(mesh=mesh, dt=dt)

            ERRORS_dX[i, j] = err

    diag = ERRORS_dX.diagonal()
    rates = np.log(diag[:-1]/diag[1:])/np.log(true_hs[:-1]/true_hs[1:])
    print(rates)
    assert all(0.9 <= num for num in rates)