"""Convergence study: Advection-diffusion-reaction convergence tests on surfaces."""

import numpy as np
import pytest
from ngsolve import *

from cosmos.core.model import CosmosModel
from cosmos.pde import (
    ADRBoundarySystemBDF1Model,
    ADRBoundarySystemBDF1StabModel,
)
from cosmos.utils.generate_meshes import generate_boundary_half_sphere, generate_boundary_sphere
from cosmos.utils.tools import gradient

pytestmark = pytest.mark.convergence
pytestmark = pytest.mark.slow


def _build(solver, dt, maxh, b, c, d):

    mesh = generate_boundary_half_sphere(maxh=maxh, R=1.0)
    t = Parameter(0.0)
    model = CosmosModel("adr", mesh, t0=0.0, t1=1.0, dt=dt, t=t, coupling_type="explicit")
    compartment = model.create_compartment("surface", boundary="default", bboundary="bboundary")
    pde = model.create_pde("adr", solver, compartment, ale_type=1, dim=1)
    u_ex, rhs, displacement_ex, P_ex = _build_manufactured_solutions(dt, t, b, c, d)
    pde.set_params(
        u0_1=u_ex,
        c_1=c,
        b_1=lambda: model.ale.V + b,
        d_1=d,
        rhs_1=rhs,
        u_bnd_1=u_ex,
        gradu_bnd_1=gradient(u_ex, P_ex),
    )
    ale = model.create_ale("ale", compartment=compartment)
    ale.set_domain_velocity(displacement_ex / dt)
    return model, pde, u_ex


def _build_manufactured_solutions(dt, t, b, c, d):
    """Build a set of manufactured solutions for the ADR system."""

    def A_B(t):
        A = CF((cos(t), -sin(t), 0, sin(t), cos(t), 0, 0, 0, 1), dims=(3, 3))
        B = CF((0, 0, t))
        return A, B

    A, B = A_B(t)
    detJ = Det(A)
    invA = Cof(A).trans / detJ
    inv_phi = invA * (CF((x, y, z)) - B)  # Inverse of deformation map
    w_phi = A.Diff(t) * inv_phi + B.Diff(t)  # velocity of the moving domain
    n_ex = Cof(A) * inv_phi / Norm(Cof(A) * inv_phi)
    P_ex = Id(3) - OuterProduct(n_ex, n_ex)
    b = P_ex * b
    Ap, Bp = A_B(t + dt)
    displacement_ex = (Ap - A) * inv_phi + (Bp - B)

    u_ex = sin(2 * pi * x) * sin(2 * t)  # Exact solution
    flux = b * u_ex - d * gradient(u_ex, P_ex)  # total flux
    rel_flux = u_ex * Trace(gradient(displacement_ex / dt, P_ex)) + displacement_ex / dt * gradient(
        u_ex, Id(3)
    )
    rhs = (
        u_ex.Diff(t) + rel_flux + Trace(gradient(flux, P_ex)) + c * u_ex
    ).Compile()  # manufactured solution right-hand side

    return u_ex, rhs, displacement_ex, P_ex


def _generate_convergence_array(solver, b, c, d, **params):
    """Generate the error matrices related to a certain convergence study."""

    power_t = 1.5
    dt0 = 0.1
    dt_refs = 3
    dts = dt0 / (power_t ** (np.arange(dt_refs + 1)))

    power_h = 1.5
    dh0 = 0.2
    dh_refs = 3
    dhs = dh0 / (power_h ** (np.arange(dh_refs + 1)))

    ERRORS_ADR = np.zeros((len(dts), len(dhs)))
    ERRORS_ADR.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):
            model, pde, u_ex = _build(solver, dt=dt, maxh=dh, b=b, c=c, d=d)
            pde.set_params(**params)

            def error():
                return sqrt(
                    Integrate(
                        InnerProduct(pde.sol[0] - u_ex, pde.sol[0] - u_ex), model.parentmesh, BND
                    )
                )

            errs = []
            for _ in model():
                errs.append(error())
            true_h = GridFunction(SurfaceL2(model.parentmesh, order=0))
            true_h.Set(specialcf.mesh_size, definedon=model.parentmesh.Boundaries(".*"))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h
            err = np.sqrt(dt * np.sum(np.array(errs) ** 2))
            ERRORS_ADR[i, j] = err

    return true_hs, dts, ERRORS_ADR


@pytest.mark.parametrize("solver", [ADRBoundarySystemBDF1Model, ADRBoundarySystemBDF1StabModel])
def test_convergence_adr_pure_reaction_problem(solver):

    c = CF(1.0)
    b = CF((0, 0, 0))
    d = CF(0.0)
    dhs, dts, ERRORS_ADR = _generate_convergence_array(solver=solver, b=b, c=c, d=d)
    diag = ERRORS_ADR.diagonal()
    rates = np.log(diag[:-1] / diag[1:]) / np.log(1.5)
    print("Rates", rates)

    assert all(0.9 <= num for num in rates[1:])


@pytest.mark.parametrize("solver", [ADRBoundarySystemBDF1Model, ADRBoundarySystemBDF1StabModel])
@pytest.mark.parametrize("bnd_cond", ["dir", "neu"])
def test_convergence_adr_pure_diffusion_problem(solver, bnd_cond):

    c = CF(0.0)
    b = CF((0, 0, 0))
    d = CF(1.0)
    if bnd_cond == "dir":
        dhs, dts, ERRORS_ADR = _generate_convergence_array(
            solver=solver, b=b, c=c, d=d, Dir_bnd="bboundary"
        )
    elif bnd_cond == "neu":
        dhs, dts, ERRORS_ADR = _generate_convergence_array(
            solver=solver, b=b, c=c, d=d, Neu_bnd="bboundary"
        )
    diag = ERRORS_ADR.diagonal()
    rates = np.log(diag[:-1] / diag[1:]) / np.log(1.5)
    print("Rates", rates)

    assert all(0.9 <= num for num in rates[1:])


@pytest.mark.parametrize("solver", [ADRBoundarySystemBDF1Model, ADRBoundarySystemBDF1StabModel])
def test_convergence_adr_pure_advection_problem(solver):

    c = CF(0.0)
    b = CF((2, 1, 0))
    d = CF(0.0)
    dhs, dts, ERRORS_ADR = _generate_convergence_array(solver=solver, b=b, c=c, d=d)
    diag = ERRORS_ADR.diagonal()
    rates = np.log(diag[:-1] / diag[1:]) / np.log(dhs[:-1] / dhs[1:])
    print(ERRORS_ADR)
    print("Rates", rates)

    assert all(0.9 <= num for num in rates[1:])


@pytest.mark.parametrize("solver", [ADRBoundarySystemBDF1Model, ADRBoundarySystemBDF1StabModel])
@pytest.mark.parametrize("bnd_cond", ["dir", "neu"])
def test_convergence_adr_bounded_problem(solver, bnd_cond):

    c = CF(1.0)
    b = CF((2, 1, 0))
    d = CF(1.0)
    if bnd_cond == "dir":
        dhs, dts, ERRORS_ADR = _generate_convergence_array(
            solver=solver, b=b, c=c, d=d, Dir_bnd="bboundary", bounds_1=[-1, 1]
        )
    elif bnd_cond == "neu":
        dhs, dts, ERRORS_ADR = _generate_convergence_array(
            solver=solver, b=b, c=c, d=d, Neu_bnd="bboundary", bounds_1=[-1, 1]
        )
    diag = ERRORS_ADR.diagonal()
    rates = np.log(diag[:-1] / diag[1:]) / np.log(dhs[:-1] / dhs[1:])
    print("Rates", rates)

    assert all(0.9 <= num for num in rates[1:])


@pytest.mark.parametrize("solver", [ADRBoundarySystemBDF1Model, ADRBoundarySystemBDF1StabModel])
@pytest.mark.parametrize("bnd_cond", ["dir", "neu"])
def test_convergence_adr_mass_preserving_problem(solver, bnd_cond):

    c = CF(1.0)
    b = CF((2, 1, 0))
    d = CF(1.0)
    if bnd_cond == "dir":
        dhs, dts, ERRORS_ADR = _generate_convergence_array(
            solver=solver, b=b, c=c, d=d, Dir_bnd="bboundary", mass_preserving_1=True
        )
    elif bnd_cond == "neu":
        dhs, dts, ERRORS_ADR = _generate_convergence_array(
            solver=solver, b=b, c=c, d=d, Neu_bnd="bboundary", mass_preserving_1=True
        )
    diag = ERRORS_ADR.diagonal()
    rates = np.log(diag[:-1] / diag[1:]) / np.log(dhs[:-1] / dhs[1:])
    print("Rates", rates)

    assert all(0.9 <= num for num in rates[1:])


@pytest.mark.parametrize("solver", [ADRBoundarySystemBDF1Model, ADRBoundarySystemBDF1StabModel])
@pytest.mark.parametrize("bnd_cond", ["dir", "neu"])
def test_convergence_adr_bounded_and_mass_preserving_problem(solver, bnd_cond):

    c = CF(1.0)
    b = CF((2, 1, 0))
    d = CF(1.0)
    if bnd_cond == "dir":
        dhs, dts, ERRORS_ADR = _generate_convergence_array(
            solver=solver,
            b=b,
            c=c,
            d=d,
            Dir_bnd="bboundary",
            bounds_1=[-1, 1],
            mass_preserving_1=True,
        )
    elif bnd_cond == "neu":
        dhs, dts, ERRORS_ADR = _generate_convergence_array(
            solver=solver,
            b=b,
            c=c,
            d=d,
            Neu_bnd="bboundary",
            bounds_1=[-1, 1],
            mass_preserving_1=True,
        )
    diag = ERRORS_ADR.diagonal()
    rates = np.log(diag[:-1] / diag[1:]) / np.log(dhs[:-1] / dhs[1:])
    print("Rates", rates)

    assert all(0.9 <= num for num in rates[1:])
