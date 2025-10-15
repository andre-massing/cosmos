from ngsolve import *
from cosmos import *
from ngsolve.webgui import Draw
from cosmos.utils.generate_meshes import generate_boundary_sphere, generate_boundary_half_sphere, \
    generate_boundary_torus, generate_boundary_half_torus
from scipy.integrate import solve_ivp
import numpy as np
import logging
import pytest
import pandas as pd

logging.getLogger().setLevel(logging.INFO)


@pytest.mark.parametrize("redistribute", [None, 'DuanLi', 'MDR'])
def test_convergence_sphere(
        request,
        artifacts_path,
        redistribute):
    
    out = artifacts_path
    filename = request.function.__name__

    Tend = 1
    def solve(mesh, dt):

        gfu = GridFunction(H1(mesh, order = 1, definedon = mesh.Boundaries('.*')))
        ns = specialcf.normal(3)

        solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
        solvermesh = SolverMesh(mesh)
        solver = Solver(solvermesh, solvertime, printing=True)

        
        ale = ALEModel(solver, 2)
        if redistribute:
            pde = WillmoreBoundaryBDF1Model(solver, 1, input_params={'autoupdate': True})
            ale.set_bnd_displacement(pde.displacement, domain='default', 
                                     redistribute=True, redistribute_type=redistribute)
        else:
            pde = WillmoreBoundaryBDF1Model(solver, 1)
            ale.set_bnd_displacement(pde.displacement, domain='default')

        errs = []
        for _ in solver():
            mesh.deformation.vec.data = solver.mesh.prev_deformation[-2].vec.data
            gfu.Set(pde.gfu_D*ns, dual = True, definedon = mesh.Boundaries('.*'))
            mesh.deformation.vec.data =  solver.mesh.prev_deformation[-1].vec.data
            errs.append(np.max(np.abs(gfu.vec.FV().NumPy())))

        return np.max(np.array(errs))

    power_t = 1.5
    dt0 = 0.1
    dt_refs = 3
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h = 1.5
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
    dhs = true_hs

    os.makedirs(out, exist_ok=True)

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_dX))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out,'sphere_stationary_th.dat'), sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_dX.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out, 'sphere_stationary_ht.dat'), sep='\t', index=False)

    assert 1

@pytest.mark.parametrize("redistribute", [None, 'DuanLi', 'MDR'])
def test_convergence_half_sphere(
        request,
        artifacts_path,
        redistribute):
    
    out = artifacts_path
    filename = request.function.__name__

    Tend = 1
    def solve(mesh, dt):

        gfu = GridFunction(H1(mesh, order = 1, definedon = mesh.Boundaries('.*')))
        ns = specialcf.normal(3)

        solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
        solvermesh = SolverMesh(mesh)
        solver = Solver(solvermesh, solvertime, printing=True)

        ale = ALEModel(solver, 2)
        if redistribute:
            input_params = {
                "clamped_bnd": 'bboundary',
                "clamped_conormal": CF((0, 0, -1)),
                "autoupdate": True
            }
            pde = WillmoreBoundaryBDF1Model(solver, 1, input_params=input_params)
            ale.set_bnd_displacement(pde.displacement, domain='default', clamped_bnd='bboundary',
                                     redistribute=True, redistribute_type=redistribute)
        else:
            input_params = {
                "clamped_bnd": 'bboundary',
                "clamped_conormal": CF((0, 0, -1))
            }
            pde = WillmoreBoundaryBDF1Model(solver, 1, input_params=input_params)
            ale.set_bnd_displacement(pde.displacement, domain='default', clamped_bnd='bboundary')

        errs = []
        for _ in solver():
            mesh.deformation.vec.data = solver.mesh.prev_deformation[-2].vec.data
            gfu.Set(pde.gfu_D*ns, dual = True, definedon = mesh.Boundaries('.*'))
            mesh.deformation.vec.data =  solver.mesh.prev_deformation[-1].vec.data
            errs.append(np.max(np.abs(gfu.vec.FV().NumPy())))

        return np.max(np.array(errs))

    power_t = 1.5
    dt0 = 0.1
    dt_refs = 3
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h = 1.5
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
    dhs = true_hs

    os.makedirs(out, exist_ok=True)

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_dX))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out,'half_sphere_stationary_th.dat'), sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_dX.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out, 'half_sphere_stationary_ht.dat'), sep='\t', index=False)

    assert 1

@pytest.mark.parametrize("redistribute", [None, 'DuanLi', 'MDR'])
def test_convergence_torus(
        request,
        artifacts_path,
        redistribute):
    
    out = artifacts_path
    filename = request.function.__name__

    Tend = 1
    def solve(mesh, dt):

        gfu = GridFunction(H1(mesh, order = 1, definedon = mesh.Boundaries('.*')))
        ns = specialcf.normal(3)

        solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
        solvermesh = SolverMesh(mesh)
        solver = Solver(solvermesh, solvertime, printing=True)

        
        ale = ALEModel(solver, 2)
        if redistribute:
            pde = WillmoreBoundaryBDF1Model(solver, 1, input_params={'autoupdate': True})
            ale.set_bnd_displacement(pde.displacement, domain='default', 
                                     redistribute=True, redistribute_type=redistribute)
        else:
            pde = WillmoreBoundaryBDF1Model(solver, 1)
            ale.set_bnd_displacement(pde.displacement, domain='default')

        errs = []
        for _ in solver():
            mesh.deformation.vec.data = solver.mesh.prev_deformation[-2].vec.data
            gfu.Set(pde.gfu_D*ns, dual = True, definedon = mesh.Boundaries('.*'))
            mesh.deformation.vec.data =  solver.mesh.prev_deformation[-1].vec.data
            errs.append(np.max(np.abs(gfu.vec.FV().NumPy())))

        return np.max(np.array(errs))

    power_t = 1.5
    dt0 = 0.1
    dt_refs = 3
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h = 1.5
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
    dhs = true_hs

    os.makedirs(out, exist_ok=True)

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_dX))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out,'torus_stationary_th.dat'), sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_dX.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out, 'torus_stationary_ht.dat'), sep='\t', index=False)

    assert 1

@pytest.mark.parametrize("redistribute", [None, 'DuanLi', 'MDR'])
def test_convergence_half_torus(
        request,
        artifacts_path,
        redistribute):
    
    out = artifacts_path
    filename = request.function.__name__

    Tend = 1
    def solve(mesh, dt):

        gfu = GridFunction(H1(mesh, order = 1, definedon = mesh.Boundaries('.*')))
        ns = specialcf.normal(3)

        solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
        solvermesh = SolverMesh(mesh)
        solver = Solver(solvermesh, solvertime, printing=True)

        
        ale = ALEModel(solver, 2)
        if redistribute:
            input_params = {
                "clamped_bnd": 'bboundary',
                "clamped_conormal": CF((0, 0, -1)),
                "autoupdate": True
            }
            pde = WillmoreBoundaryBDF1Model(solver, 1, input_params=input_params)
            ale.set_bnd_displacement(pde.displacement, domain='default', 
                                     redistribute=True, redistribute_type=redistribute,
                                     clamped_bnd='bboundary')
        else:
            input_params = {
                "clamped_bnd": 'bboundary',
                "clamped_conormal": CF((0, 0, -1)),
            }
            pde = WillmoreBoundaryBDF1Model(solver, 1, input_params=input_params)
            ale.set_bnd_displacement(pde.displacement, domain='default', clamped_bnd='bboundary')

        errs = []
        for _ in solver():
            mesh.deformation.vec.data = solver.mesh.prev_deformation[-2].vec.data
            gfu.Set(pde.gfu_D*ns, dual = True, definedon = mesh.Boundaries('.*'))
            mesh.deformation.vec.data =  solver.mesh.prev_deformation[-1].vec.data
            errs.append(np.max(np.abs(gfu.vec.FV().NumPy())))

        return np.max(np.array(errs))

    power_t = 1.5
    dt0 = 0.1
    dt_refs = 3
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h = 1.5
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
    dhs = true_hs

    os.makedirs(out, exist_ok=True)

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_dX))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out,'half_torus_stationary_th.dat'), sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_dX.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out, 'half_torus_stationary_ht.dat'), sep='\t', index=False)

    assert 1


@pytest.mark.parametrize("redistribute", [None, 'DuanLi', 'MDR'])
def test_convergence_sphere_kappa0(
        request,
        artifacts_path,
        redistribute):
    
    out = artifacts_path
    filename = request.function.__name__

    sp_curv = -1
    R0 = 1
    Tend = 1
    F = lambda t, s: -sp_curv/s*(2/s+sp_curv)
    t_eval = np.arange(0, Tend, Tend/1000)
    sol = solve_ivp(F, [0, Tend], [R0], t_eval=t_eval, dense_output=True)

    Tend = 1
    def solve(mesh, dt):

        gfu = GridFunction(H1(mesh, order = 1, definedon = mesh.Boundaries('.*')))
        ns = specialcf.normal(3)

        solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
        solvermesh = SolverMesh(mesh)
        solver = Solver(solvermesh, solvertime, printing=True)

        
        ale = ALEModel(solver, 2)
        if redistribute:
            pde = WillmoreBoundaryBDF1Model(solver, 1, input_params={'autoupdate': True})
            ale.set_bnd_displacement(pde.displacement, domain='default', 
                                     redistribute=True, redistribute_type=redistribute)
        else:
            pde = WillmoreBoundaryBDF1Model(solver, 1)
            ale.set_bnd_displacement(pde.displacement, domain='default')
        pde.set_input_fields({
            "spontaneous_curvature": sp_curv, 
        })

        errs = []
        for _ in solver():
            Rt = sol.sol(solver.current_time+dt)
            gfu.Set(sqrt(x**2+y**2+z**2), dual = True, definedon = mesh.Boundaries('.*'))
            errs.append(np.max(np.abs(gfu.vec.FV().NumPy() - Rt)))

        return np.max(np.array(errs))

    power_t = 1.5
    dt0 = 0.002
    dt_refs = 3
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h = 1.5
    dh0 = 0.3
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
    dhs = true_hs

    os.makedirs(out, exist_ok=True)

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_dX))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out,'sphere_kappa0_th.dat'), sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_dX.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out, 'sphere_kappa0_ht.dat'), sep='\t', index=False)

    assert 1
