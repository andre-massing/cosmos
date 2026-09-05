"""Convergence study: Advection-diffusion-reaction convergence tests on bulk domains.
"""

import numpy as np
import pytest
from ngsolve import *
from cosmos.core.model import CosmosModel
from cosmos.pde import ADRVolumeSystemBDF1Model
from cosmos.utils.generate_meshes import generate_volume_circle
from cosmos.utils.tools import gradient

pytestmark = pytest.mark.convergence

def _build(dt, maxh, b, c, d):

    mesh = generate_volume_circle(maxh=maxh, R=1.0)
    t = Parameter(0.0)
    model = CosmosModel("adr", mesh, t0=0.0, t1=1.0, dt=dt, t=t, coupling_type="explicit")
    compartment = model.create_compartment("bulk", material = "default", boundary="boundary")
    pde = model.create_pde("adr", ADRVolumeSystemBDF1Model, compartment, ale_type=1, dim=1)
    u_ex, rhs, displacement_ex = _build_manufactured_solutions(dt, t, b, c, d)
    pde.set_params(
        u0_1=u_ex,
        c_1=c,
        b_1= lambda: model.ale.V + b,
        d_1=d,
        rhs_1=rhs,
        u_bnd_1=u_ex,
        gradu_bnd_1=gradient(u_ex, Id(2))
    )
    ale = model.create_ale("ale", compartment=compartment)
    ale.set_domain_velocity(displacement_ex / dt)
    return model, pde, u_ex

def _build_manufactured_solutions(dt, t, b, c, d):
    """Build a set of manufactured solutions for the ADR system."""

    def A_B(t):
        A = CF((cos(t), -sin(t), sin(t), cos(t)), dims=(2, 2))
        B = CF((0, 2*t))
        return A, B
    A, B = A_B(t)
    detJ = Det(A)
    invA = Cof(A).trans / detJ
    inv_phi = invA * (CF((x, y)) - B)  # Inverse of deformation map
    w_phi = A.Diff(t) * inv_phi + B.Diff(t)  # velocity of the moving domain
    Ap, Bp = A_B(t + dt)
    displacement_ex = (Ap - A) * inv_phi + (Bp - B)
    
    u_ex = sin(2 * pi * x)*sin(2*t)  # Exact solution
    flux = (b + w_phi) * u_ex - d * gradient(u_ex, Id(2))  # total flux
    rhs = (
        u_ex.Diff(t) + Trace(gradient(flux, Id(2))) + c * u_ex
    )  # manufactured solution right-hand side

    return u_ex, rhs, displacement_ex

def _generate_convergence_array(b, c, d, **params):
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

            model, pde, u_ex = _build(dt=dt, maxh=dh, b=b, c=c, d=d)
            pde.set_params(**params)

            def error():
                return sqrt(Integrate(InnerProduct(pde.sol[0] - u_ex, pde.sol[0] - u_ex), model.parentmesh, VOL))
            errs = []
            for _ in model():
                errs.append(error())
            true_h = GridFunction(L2(model.parentmesh, order=0))
            true_h.Set(specialcf.mesh_size)
            true_hs[j] = np.max(true_h.vec.data)
            err = np.sqrt(dt * np.sum(np.array(errs) ** 2))
            ERRORS_ADR[i, j] = err
    
    return true_hs, dts, ERRORS_ADR

def test_convergence_adr_pure_reaction_problem():

    c = CF(1.0)
    b = CF((0, 0))
    d = CF(0.0)
    dhs, dts, ERRORS_ADR = _generate_convergence_array(b=b, c=c, d=d)
    diag = ERRORS_ADR.diagonal()
    rates = np.log(diag[:-1]/diag[1:])/np.log(1.5)
    print("Rates", rates)

    assert all(0.9 <= num for num in rates)

def test_convergence_adr_pure_diffusion_problem():

    c = CF(0.0)
    b = CF((0, 0))
    d = CF(1.0)
    dhs, dts, ERRORS_ADR = _generate_convergence_array(b=b, c=c, d=d, Neu_bnd = "boundary")
    diag = ERRORS_ADR.diagonal()
    rates = np.log(diag[:-1]/diag[1:])/np.log(1.5)
    print("Rates", rates)

    assert all(0.9 <= num for num in rates)

def test_convergence_adr_pure_advection_problem():

    c = CF(0.0)
    b = CF((2, 1))
    d = CF(0.0)
    dhs, dts, ERRORS_ADR = _generate_convergence_array(b=b, c=c, d=d)
    diag = ERRORS_ADR.diagonal()
    rates = np.log(diag[:-1]/diag[1:])/np.log(dhs[:-1]/dhs[1:])
    print(ERRORS_ADR)
    print("Rates", rates)

    assert all(0.9 <= num for num in rates)

def test_convergence_adr_bounded_problem():

    c = CF(1.0)
    b = CF((2, 1))
    d = CF(1.0)
    dhs, dts, ERRORS_ADR = _generate_convergence_array(b=b, c=c, d=d, Neu_bnd = "boundary", bounds_1 = [-1, 1])
    diag = ERRORS_ADR.diagonal()
    rates = np.log(diag[:-1]/diag[1:])/np.log(1.5)
    print("Rates", rates)

    assert all(0.9 <= num for num in rates)

def test_convergence_adr_mass_preserving_problem():

    c = CF(1.0)
    b = CF((2, 1))
    d = CF(1.0)
    dhs, dts, ERRORS_ADR = _generate_convergence_array(b=b, c=c, d=d, Neu_bnd = "boundary", mass_preserving_1 = True)
    diag = ERRORS_ADR.diagonal()
    rates = np.log(diag[:-1]/diag[1:])/np.log(dhs[:-1]/dhs[1:])
    print("Rates", rates)

    assert all(0.9 <= num for num in rates)

def test_convergence_adr_bounded_and_mass_preserving_problem():

    c = CF(1.0)
    b = CF((2, 1))
    d = CF(1.0)
    dhs, dts, ERRORS_ADR = _generate_convergence_array(b=b, c=c, d=d, Neu_bnd = "boundary", bounds_1 = [-1, 1], mass_preserving_1 = True)
    diag = ERRORS_ADR.diagonal()
    rates = np.log(diag[:-1]/diag[1:])/np.log(dhs[:-1]/dhs[1:])
    print("Rates", rates)

    assert all(0.9 <= num for num in rates)