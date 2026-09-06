"""Convergence study: Gradient flow convergence tests.
"""

import numpy as np
import pytest
from ngsolve import *
from cosmos import *
from ngsolve.webgui import Draw
from cosmos.utils.generate_meshes import (
    generate_boundary_sphere,
    generate_boundary_half_sphere,
    generate_boundary_torus,
    generate_boundary_half_torus,
)
from cosmos.core.model import CosmosModel
from cosmos.pde import (
    GeometricalFlowModel,
    GeometricalFlowStationaryModel
)
from scipy.integrate import solve_ivp

pytestmark = pytest.mark.convergence
pytestmark = pytest.mark.slow

@pytest.mark.parametrize("redistribute", [False, True])
def test_convergence_willmore_flow_sphere(redistribute):

    Tend = 1
    def solve(mesh, dt):

        model = CosmosModel(name = 'convergence_willmore_flow_sphere', parentmesh=mesh,
                            t0 = 0, t1 = Tend, dt = dt, redistribute = redistribute,
                            coupling_type = 'implicit')

        comp1 = model.create_compartment(name = 'compartment', boundary = 'default', bboundary = '')
        pde = model.create_pde(name = 'willmore_flow', pde_model=GeometricalFlowModel, compartment=comp1, ale_type = 0)
        ale = model.create_ale(name = 'ale', compartment=comp1)
        ale.set_normal_velocity(pde.V_h)
        ale.set_tangential_velocity(CF((0, 0, 0)))

        errs = []
        for _ in model():
            errs.append(np.max(np.abs(dt*ale.gfu_norm_vel.vec.FV().NumPy())))

        return np.max(np.array(errs))

    power_t = 1.5
    dt0 = 0.1
    dt_refs = 3
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h= 1.5
    dh0 = 0.2
    dh_refs = 3
    dhs = dh0/(power_h**(np.arange(dh_refs+1)))

    ERRORS_dX = np.zeros((len(dts), len(dhs)))
    ERRORS_dX.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):

            mesh = generate_boundary_sphere(maxh = dh, R = 1)

            true_h = GridFunction(SurfaceL2(mesh, order = 0))
            true_h.Set(get_config().h, definedon = mesh.Boundaries('.*'))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            err = solve(mesh=mesh, dt=dt)

            ERRORS_dX[i, j] = err

    diag = ERRORS_dX.diagonal()
    rates = np.log(diag[:-1]/diag[1:])/np.log(true_hs[:-1]/true_hs[1:])
    print(rates)
    assert all(0.9 <= num for num in rates)

@pytest.mark.parametrize("redistribute", [False, True])
def test_convergence_willmore_flow_half_sphere(redistribute):

    Tend = 1
    def solve(mesh, dt):

        model = CosmosModel(name = 'convergence_willmore_flow_half_sphere', parentmesh=mesh,
                            t0 = 0, t1 = Tend, dt = dt, redistribute = redistribute,
                            coupling_type = 'implicit')

        comp1 = model.create_compartment(name = 'compartment', boundary = 'default', bboundary = 'bboundary',
                                         clamped_bbnd = 'bboundary')
        pde = model.create_pde(name = 'willmore_flow', pde_model=GeometricalFlowModel, compartment=comp1, ale_type = 0)
        ale = model.create_ale(name = 'ale', compartment=comp1)
        ale.set_normal_velocity(pde.V_h)
        ale.set_tangential_velocity(CF((0, 0, 0)))

        errs = []
        for _ in model():
            errs.append(np.max(np.abs(dt*ale.gfu_norm_vel.vec.FV().NumPy())))

        return np.max(np.array(errs))

    power_t = 1.5
    dt0 = 0.1
    dt_refs = 3
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h= 1.5
    dh0 = 0.2
    dh_refs = 3
    dhs = dh0/(power_h**(np.arange(dh_refs+1)))

    ERRORS_dX = np.zeros((len(dts), len(dhs)))
    ERRORS_dX.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):

            mesh = generate_boundary_half_sphere(maxh = dh, R = 1)

            true_h = GridFunction(SurfaceL2(mesh, order = 0))
            true_h.Set(get_config().h, definedon = mesh.Boundaries('.*'))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            err = solve(mesh=mesh, dt=dt)

            ERRORS_dX[i, j] = err

    diag = ERRORS_dX.diagonal()
    rates = np.log(diag[:-1]/diag[1:])/np.log(true_hs[:-1]/true_hs[1:])
    print(rates)
    assert all(0.9 <= num for num in rates)

@pytest.mark.parametrize("redistribute", [False, True])
def test_convergence_willmore_flow_clifford_torus(redistribute):

    Tend = 1
    def solve(mesh, dt):

        model = CosmosModel(name = 'convergence_willmore_flow_clifford_torus', parentmesh=mesh,
                            t0 = 0, t1 = Tend, dt = dt, redistribute = redistribute,
                            coupling_type = 'implicit')

        comp1 = model.create_compartment(name = 'compartment', boundary = 'default', bboundary = '')
        pde = model.create_pde(name = 'willmore_flow', pde_model=GeometricalFlowModel, compartment=comp1, ale_type = 0)
        ale = model.create_ale(name = 'ale', compartment=comp1)
        ale.set_normal_velocity(pde.V_h)
        ale.set_tangential_velocity(CF((0, 0, 0)))

        errs = []
        for _ in model():
            errs.append(np.max(np.abs(dt*ale.gfu_norm_vel.vec.FV().NumPy())))

        return np.max(np.array(errs))

    power_t = 1.5
    dt0 = 0.1
    dt_refs = 3
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h= 1.5
    dh0 = 0.2
    dh_refs = 3
    dhs = dh0/(power_h**(np.arange(dh_refs+1)))

    ERRORS_dX = np.zeros((len(dts), len(dhs)))
    ERRORS_dX.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):

            mesh = generate_boundary_torus(maxh = dh, R = sqrt(2), r=1)

            true_h = GridFunction(SurfaceL2(mesh, order = 0))
            true_h.Set(get_config().h, definedon = mesh.Boundaries('.*'))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            err = solve(mesh=mesh, dt=dt)

            ERRORS_dX[i, j] = err

    diag = ERRORS_dX.diagonal()
    rates = np.log(diag[:-1]/diag[1:])/np.log(true_hs[:-1]/true_hs[1:])
    print(rates)
    assert all(0.9 <= num for num in rates)

@pytest.mark.parametrize("redistribute", [False, True])
def test_convergence_willmore_flow_half_clifford_torus(redistribute):

    Tend = 1
    def solve(mesh, dt):

        model = CosmosModel(name = 'convergence_willmore_flow_half_clifford_torus', parentmesh=mesh,
                            t0 = 0, t1 = Tend, dt = dt, redistribute = redistribute,
                            coupling_type = 'implicit')

        comp1 = model.create_compartment(name = 'compartment', boundary = 'default', bboundary = 'bboundary',
                                         clamped_bbnd = 'bboundary')
        pde = model.create_pde(name = 'willmore_flow', pde_model=GeometricalFlowModel, compartment=comp1, ale_type = 0)
        ale = model.create_ale(name = 'ale', compartment=comp1)
        ale.set_normal_velocity(pde.V_h)
        ale.set_tangential_velocity(CF((0, 0, 0)))

        errs = []
        for _ in model():
            errs.append(np.max(np.abs(dt*ale.gfu_norm_vel.vec.FV().NumPy())))

        return np.max(np.array(errs))

    power_t = 1.5
    dt0 = 0.1
    dt_refs = 3
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h= 1.5
    dh0 = 0.2
    dh_refs = 3
    dhs = dh0/(power_h**(np.arange(dh_refs+1)))

    ERRORS_dX = np.zeros((len(dts), len(dhs)))
    ERRORS_dX.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):

            mesh = generate_boundary_half_torus(maxh = dh, R = sqrt(2), r=1)

            true_h = GridFunction(SurfaceL2(mesh, order = 0))
            true_h.Set(get_config().h, definedon = mesh.Boundaries('.*'))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            err = solve(mesh=mesh, dt=dt)

            ERRORS_dX[i, j] = err

    diag = ERRORS_dX.diagonal()
    rates = np.log(diag[:-1]/diag[1:])/np.log(true_hs[:-1]/true_hs[1:])
    print(rates)
    assert all(0.9 <= num for num in rates)

@pytest.mark.parametrize("redistribute", [False, True])
def test_convergence_willmore_flow_sphere_with_spontaneous_curvature(redistribute):


    sp_curv = -1
    R0 = 1
    Tend = 1
    F = lambda t, s: -sp_curv / s * (2 / s + sp_curv)
    t_eval = np.arange(0, Tend, Tend / 1000)
    sol = solve_ivp(F, [0, Tend], [R0], t_eval=t_eval, dense_output=True)

    Tend = 1

    def solve(mesh, dt):

        gfu = GridFunction(H1(mesh))

        model = CosmosModel(
            name="convergence_willmore_flow_sphere_with_spontaneous_curvature",
            parentmesh=mesh,
            t0=0,
            t1=Tend,
            dt=dt,
            redistribute=redistribute,
            coupling_type="explicit"
        )

        comp1 = model.create_compartment(name="compartment", boundary="default", bboundary="")
        pde = model.create_pde(
            name="willmore_flow", pde_model=GeometricalFlowModel, compartment=comp1, ale_type=0
        )
        pde.set_params(sp_curv=CF(sp_curv))
        ale = model.create_ale(name="ale", compartment=comp1)
        ale.set_normal_velocity(pde.V_h)
        ale.set_tangential_velocity(CF((0, 0, 0)))

        errs = []
        for _ in model():
            Rt = sol.sol(model.t.Get())
            gfu.Set(sqrt(x**2 + y**2 + z**2), dual=True, definedon=mesh.Boundaries(".*"))
            errs.append(np.max(np.abs(gfu.vec.FV().NumPy() - Rt)))

        return np.max(np.array(errs))

    power_t = 1.5
    dt0 = 0.1
    dt_refs = 3
    dts = dt0 / (power_t ** (np.arange(dt_refs + 1)))

    power_h= 1.5
    dh0 = 0.2
    dh_refs = 3
    dhs = dh0 / (power_h ** (np.arange(dh_refs + 1)))

    ERRORS_dX = np.zeros((len(dts), len(dhs)))
    ERRORS_dX.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):
            mesh = generate_boundary_sphere(maxh=dh, R=1)

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


@pytest.mark.parametrize("redistribute", [False, True])
def test_convergence_willmore_flow_sphere_with_evolving_spontaneous_curvature(redistribute):

    sp_curv = -1
    R0 = 1
    Tend = 1
    F = lambda t, s: -sp_curv / s * (2 / s + sp_curv)
    t_eval = np.arange(0, Tend, Tend / 1000)
    sol = solve_ivp(F, [0, Tend], [R0], t_eval=t_eval, dense_output=True)

    Tend = 1

    def solve(mesh, dt):

        gfu = GridFunction(H1(mesh))

        model = CosmosModel(
            name="convergence_torus",
            parentmesh=mesh,
            t0=0,
            t1=Tend,
            dt=dt,
            redistribute=redistribute,
            coupling_type="explicit",
        )

        comp1 = model.create_compartment(name="compartment", boundary="default", bboundary="")
        pde = model.create_pde(
            name="convergence_willmore_flow_sphere_with_evolving_spontaneous_curvature",
            pde_model=GeometricalFlowStationaryModel,
            compartment=comp1,
            ale_type=0,
        )
        pde.set_params(kappa0=CF(sp_curv))
        ale = model.create_ale(name="ale", compartment=comp1)
        ale.set_normal_velocity(pde.V_h)
        ale.set_tangential_velocity(CF((0, 0, 0)))

        errs = []
        for _ in model():
            Rt = sol.sol(model.t.Get())
            gfu.Set(sqrt(x**2 + y**2 + z**2), dual=True, definedon=mesh.Boundaries(".*"))
            errs.append(np.max(np.abs(gfu.vec.FV().NumPy() - Rt)))

        return np.max(np.array(errs))

    power_t = 1.5
    dt0 = 0.1
    dt_refs = 3
    dts = dt0 / (power_t ** (np.arange(dt_refs + 1)))

    power_h= 1.5
    dh0 = 0.2
    dh_refs = 3
    dhs = dh0 / (power_h ** (np.arange(dh_refs + 1)))

    ERRORS_dX = np.zeros((len(dts), len(dhs)))
    ERRORS_dX.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):
            mesh = generate_boundary_sphere(maxh=dh, R=1)

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
