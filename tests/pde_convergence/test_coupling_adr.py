"""Convergence study: Advection-diffusion-reaction convergence tests on coupled, nonlinear bulk-surface domains."""

import numpy as np
import pytest
from ngsolve import *

from cosmos.core.model import CosmosModel
from cosmos.pde import (
    ADRBoundarySystemBDF1Model,
    ADRBoundarySystemBDF1StabModel,
    ADRVolumeSystemBDF1Model,
)
from cosmos.utils.generate_meshes import generate_volume_circle
from cosmos.utils.tools import gradient

pytestmark = pytest.mark.convergence
pytestmark = pytest.mark.slow


def _build(solver, dt, maxh, b_vol, c_vol, d_vol, b_bnd, c_bnd, d_bnd, alpha):

    mesh = generate_volume_circle(maxh=maxh, R=1.0)
    t = Parameter(0.0)
    model = CosmosModel(
        "adr", mesh, t0=0.0, t1=1.0, dt=dt, t=t, coupling_type="implicit", root=".", samples=100
    )
    compartment_vol = model.create_compartment("bulk", material="default", boundary="boundary")
    pde_vol = model.create_pde(
        "adr_vol", ADRVolumeSystemBDF1Model, compartment_vol, ale_type=1, dim=1
    )
    compartment_bnd = model.create_compartment("surface", boundary="boundary", bboundary="")
    pde_bnd = model.create_pde("adr_bnd", solver, compartment_bnd, ale_type=1, dim=1)
    u_ex, v_ex, rhs_vol, rhs_bnd, displacement_ex, P_ex = _build_manufactured_solutions(
        dt, t, b_vol, c_vol, d_vol, b_bnd, c_bnd, d_bnd, alpha
    )

    def rhs_bnd_cf():
        return rhs_bnd + alpha * pde_vol.sol[0] ** 2 * pde_bnd.sol[0]

    pde_vol.set_params(
        u0_1=u_ex,
        c_1=c_vol,
        b_1=lambda: model.ale.V + b_vol,
        d_1=d_vol,
        rhs_1=rhs_vol,
        u_bnd_1=u_ex,
        gradu_bnd_1=gradient(u_ex, Id(2)),
        printing=True,
    )
    pde_bnd.set_params(
        u0_1=v_ex,
        c_1=c_bnd,
        b_1=lambda: model.ale.V + b_bnd,
        d_1=d_bnd,
        rhs_1=rhs_bnd_cf,
        u_bnd_1=v_ex,
        gradu_bnd_1=gradient(v_ex, P_ex),
        printing=True,
    )
    ale = model.create_ale("ale", compartment=compartment_vol)
    ale.set_domain_velocity(displacement_ex / dt)
    return model, pde_vol, pde_bnd, u_ex, v_ex


def _build_manufactured_solutions(dt, t, b_vol, c_vol, d_vol, b_bnd, c_bnd, d_bnd, alpha):
    """Build a set of manufactured solutions for the ADR system."""

    def A_B(t):
        A = CF((cos(t), -sin(t), sin(t), cos(t)), dims=(2, 2))
        B = CF((0, 2 * t))
        return A, B

    A, B = A_B(t)
    detJ = Det(A)
    invA = Cof(A).trans / detJ
    inv_phi = invA * (CF((x, y)) - B)  # Inverse of deformation map
    n_ex = Cof(A) * inv_phi / Norm(Cof(A) * inv_phi)
    P_ex = Id(2) - OuterProduct(n_ex, n_ex)
    w_phi = A.Diff(t) * inv_phi + B.Diff(t)  # velocity of the moving domain
    Ap, Bp = A_B(t + dt)
    displacement_ex = (Ap - A) * inv_phi + (Bp - B)

    u_ex = sin(2 * pi * x) * sin(2 * t)  # Exact solution on bulk domain
    v_ex = cos(2 * pi * y) * cos(2 * t)  # Exact solution on surface domain
    flux_vol = (b_vol + w_phi) * u_ex - d_vol * gradient(u_ex, Id(2))  # total flux
    rhs_vol = (
        u_ex.Diff(t) + Trace(gradient(flux_vol, Id(2))) + c_vol * u_ex
    ).Compile()  # manufactured solution right-hand side
    b_bnd = P_ex * b_bnd
    flux_bnd = b_bnd * v_ex - d_bnd * gradient(v_ex, P_ex)  # total flux
    rel_flux = v_ex * Trace(gradient(w_phi, P_ex)) + w_phi * gradient(v_ex, Id(2))
    rhs_bnd = (
        v_ex.Diff(t)
        + rel_flux
        + Trace(gradient(flux_bnd, P_ex))
        + c_bnd * v_ex
        - alpha * u_ex**2 * v_ex
    ).Compile()  # manufactured solution right-hand side

    return u_ex, v_ex, rhs_vol, rhs_bnd, displacement_ex, P_ex


def _generate_convergence_array(
    solver, b_vol, c_vol, d_vol, b_bnd, c_bnd, d_bnd, alpha, params_vol={}, params_bnd={}
):
    """Generate the error matrices related to a certain convergence study."""

    power_t = 1.5
    dt0 = 0.1
    dt_refs = 3
    dts = dt0 / (power_t ** (np.arange(dt_refs + 1)))

    power_h = 1.5
    dh0 = 0.2
    dh_refs = 3
    dhs = dh0 / (power_h ** (np.arange(dh_refs + 1)))

    ERRORS_ADR_vol = np.zeros((len(dts), len(dhs)))
    ERRORS_ADR_vol.fill(np.inf)
    ERRORS_ADR_bnd = np.zeros((len(dts), len(dhs)))
    ERRORS_ADR_bnd.fill(np.inf)

    true_hs_vol = np.zeros(len(dhs))
    true_hs_bnd = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):
            model, pde_vol, pde_bnd, u_ex, v_ex = _build(
                solver,
                dt=dt,
                maxh=dh,
                b_vol=b_vol,
                c_vol=c_vol,
                d_vol=d_vol,
                b_bnd=b_bnd,
                c_bnd=c_bnd,
                d_bnd=d_bnd,
                alpha=alpha,
            )
            pde_vol.set_params(**params_vol)
            pde_bnd.set_params(**params_bnd)

            def error_vol():
                return sqrt(
                    Integrate(
                        InnerProduct(pde_vol.sol[0] - u_ex, pde_vol.sol[0] - u_ex),
                        model.parentmesh,
                        VOL,
                    )
                )

            errs_vol = []

            def error_bnd():
                return sqrt(
                    Integrate(
                        InnerProduct(pde_bnd.sol[0] - v_ex, pde_bnd.sol[0] - v_ex),
                        model.parentmesh,
                        BND,
                    )
                )

            errs_bnd = []

            for _ in model():
                errs_vol.append(error_vol())
                errs_bnd.append(error_bnd())

            true_h_vol = GridFunction(L2(model.parentmesh, order=0))
            true_h_vol.Set(specialcf.mesh_size)
            true_hs_vol[j] = np.max(true_h_vol.vec.data)
            err_vol = np.sqrt(dt * np.sum(np.array(errs_vol) ** 2))

            true_h_bnd = GridFunction(SurfaceL2(model.parentmesh, order=0))
            true_h_bnd.Set(specialcf.mesh_size, definedon=model.parentmesh.Boundaries(".*"))
            true_h_bnd = np.max(true_h_bnd.vec.data)
            true_hs_bnd[j] = true_h_bnd
            err_bnd = np.sqrt(dt * np.sum(np.array(errs_bnd) ** 2))

            ERRORS_ADR_vol[i, j] = err_vol
            ERRORS_ADR_bnd[i, j] = err_bnd

    return true_hs_vol, true_hs_bnd, dts, ERRORS_ADR_vol, ERRORS_ADR_bnd


@pytest.mark.parametrize("solver", [ADRBoundarySystemBDF1Model, ADRBoundarySystemBDF1StabModel])
def test_convergence_coupling_adr_pure_reaction_problem(solver):

    c_vol = CF(2.0)
    b_vol = CF((0, 0))
    d_vol = CF(0.0)
    c_bnd = CF(3.0)
    b_bnd = CF((0, 0))
    d_bnd = CF(0.0)
    alpha = 1  # Coupling parameter
    dhs_vol, dhs_bnd, dts, ERRORS_ADR_vol, ERRORS_ADR_bnd = _generate_convergence_array(
        solver=solver,
        b_vol=b_vol,
        c_vol=c_vol,
        d_vol=d_vol,
        b_bnd=b_bnd,
        c_bnd=c_bnd,
        d_bnd=d_bnd,
        alpha=alpha,
    )
    diag_vol = ERRORS_ADR_vol.diagonal()
    diag_bnd = ERRORS_ADR_bnd.diagonal()
    rates_vol = np.log(diag_vol[:-1] / diag_vol[1:]) / np.log(dhs_vol[:-1] / dhs_vol[1:])
    rates_bnd = np.log(diag_bnd[:-1] / diag_bnd[1:]) / np.log(dhs_bnd[:-1] / dhs_bnd[1:])
    print("Rates vol", rates_vol)
    print("Rates bnd", rates_bnd)

    assert all(0.9 <= num for num in rates_vol)
    assert all(0.9 <= num for num in rates_bnd)


@pytest.mark.parametrize("solver", [ADRBoundarySystemBDF1Model, ADRBoundarySystemBDF1StabModel])
def test_convergence_coupling_adr_pure_diffusion_problem(solver):

    c_vol = CF(0.0)
    b_vol = CF((0, 0))
    d_vol = CF(2.0)
    c_bnd = CF(0.0)
    b_bnd = CF((0, 0))
    d_bnd = CF(3.0)
    alpha = 1  # Coupling parameter
    dhs_vol, dhs_bnd, dts, ERRORS_ADR_vol, ERRORS_ADR_bnd = _generate_convergence_array(
        solver=solver,
        b_vol=b_vol,
        c_vol=c_vol,
        d_vol=d_vol,
        b_bnd=b_bnd,
        c_bnd=c_bnd,
        d_bnd=d_bnd,
        alpha=alpha,
        params_vol={"Dir_bnd": "boundary"},
    )
    diag_vol = ERRORS_ADR_vol.diagonal()
    diag_bnd = ERRORS_ADR_bnd.diagonal()
    rates_vol = np.log(diag_vol[:-1] / diag_vol[1:]) / np.log(dhs_vol[:-1] / dhs_vol[1:])
    rates_bnd = np.log(diag_bnd[:-1] / diag_bnd[1:]) / np.log(dhs_bnd[:-1] / dhs_bnd[1:])
    print("Rates vol", rates_vol)
    print("Rates bnd", rates_bnd)

    assert all(0.9 <= num for num in rates_vol)
    assert all(0.9 <= num for num in rates_bnd)


@pytest.mark.parametrize("solver", [ADRBoundarySystemBDF1Model, ADRBoundarySystemBDF1StabModel])
def test_convergence_coupling_adr_pure_advection_problem(solver):

    c_vol = CF(0.0)
    b_vol = CF((2, 1))
    d_vol = CF(0.0)
    c_bnd = CF(0.0)
    b_bnd = CF((1, 2))
    d_bnd = CF(0.0)
    alpha = 1  # Coupling parameter
    dhs_vol, dhs_bnd, dts, ERRORS_ADR_vol, ERRORS_ADR_bnd = _generate_convergence_array(
        solver=solver,
        b_vol=b_vol,
        c_vol=c_vol,
        d_vol=d_vol,
        b_bnd=b_bnd,
        c_bnd=c_bnd,
        d_bnd=d_bnd,
        alpha=alpha,
    )
    diag_vol = ERRORS_ADR_vol.diagonal()
    diag_bnd = ERRORS_ADR_bnd.diagonal()
    rates_vol = np.log(diag_vol[:-1] / diag_vol[1:]) / np.log(dhs_vol[:-1] / dhs_vol[1:])
    rates_bnd = np.log(diag_bnd[:-1] / diag_bnd[1:]) / np.log(dhs_bnd[:-1] / dhs_bnd[1:])
    print("Rates vol", rates_vol)
    print("Rates bnd", rates_bnd)

    assert all(0.9 <= num for num in rates_vol)
    assert all(0.9 <= num for num in rates_bnd)
