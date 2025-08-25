# %%
from ngsolve import *
from cosmos import *
from cosmos.utils.generate_surface_meshes import generate_sphere, generate_half_sphere, generate_half_torus, generate_torus, generate_sigar
import os
import numpy as np
import pandas as pd
from myngspy import *
from scipy.integrate import solve_ivp

def test_sphere():

    Tend = 1
    def solve(mesh, dt):

        solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
        solvermesh = SolverMesh(mesh)
        solver = Solver(solvermesh, solvertime)

        pde = WillmoreBoundaryStabBDF1Model(solver, 1, name = 'willmore_boundary_bdf1')
        displacement = DisplacementBoundaryBDF1Coupling(solver, 2, pde.displacement)

        errs = []
        for _ in solver():
            errs.append(np.max(np.abs(pde.gfu_D.vec.FV().NumPy())))

        return np.max(np.array(errs))

    power_t = 1.5
    dt0 = 0.1
    dt_refs = 2
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h = 1.5
    dh0 = 0.2
    dh_refs = 2
    dhs = dh0/(power_h**(np.arange(dh_refs+1)))

    ERRORS_dX = np.zeros((len(dts), len(dhs)))
    ERRORS_dX.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):

            mesh, _ = generate_sphere(maxh = dh, R = 1)

            true_h = GridFunction(SurfaceL2(mesh, order = 0))
            true_h.Set(get_config().h, definedon = mesh.Boundaries('.*'))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            err = solve(mesh=mesh, dt=dt)

            ERRORS_dX[i, j] = err
    dhs = true_hs

    results_folder = './results_WillmoreBoundaryStabBDF1Model'
    os.makedirs(results_folder, exist_ok=True)

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_dX))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(results_folder + '/sphere_stationary_th.dat', sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_dX.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(results_folder + '/sphere_stationary_ht.dat', sep='\t', index=False)

def test_half_sphere():

    Tend = 1
    def solve(mesh, dt):

        solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
        solvermesh = SolverMesh(mesh)
        solver = Solver(solvermesh, solvertime)

        input_params = {
            "clamped_bnd": 'bottom',
            "clamped_conormal": CF((0, 0, -1))
        }
        pde = WillmoreBoundaryStabBDF1Model(solver, 1, name = 'willmore_boundary_bdf1', input_params=input_params)
        displacement = DisplacementBoundaryBDF1Coupling(solver, 2, pde.displacement)

        errs = []
        for _ in solver():
            errs.append(np.max(np.abs(pde.gfu_D.vec.FV().NumPy())))

        return np.max(np.array(errs))

    power_t = 1.5
    dt0 = 0.1
    dt_refs = 2
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h = 1.5
    dh0 = 0.2
    dh_refs = 2
    dhs = dh0/(power_h**(np.arange(dh_refs+1)))

    ERRORS_dX = np.zeros((len(dts), len(dhs)))
    ERRORS_dX.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):

            mesh, _ = generate_half_sphere(maxh = dh, R = 1)

            true_h = GridFunction(SurfaceL2(mesh, order = 0))
            true_h.Set(get_config().h, definedon = mesh.Boundaries('.*'))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            err = solve(mesh=mesh, dt=dt)

            ERRORS_dX[i, j] = err
    dhs = true_hs

    results_folder = './results_WillmoreBoundaryStabBDF1Model'
    os.makedirs(results_folder, exist_ok=True)

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_dX))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(results_folder + '/half_sphere_stationary_th.dat', sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_dX.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(results_folder + '/half_sphere_stationary_ht.dat', sep='\t', index=False)

def test_clifford_torus():

    Tend = 1
    def solve(mesh, dt):

        solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
        solvermesh = SolverMesh(mesh)
        solver = Solver(solvermesh, solvertime)

        pde = WillmoreBoundaryStabBDF1Model(solver, 1, name = 'willmore_boundary_bdf1')
        displacement = DisplacementBoundaryBDF1Coupling(solver, 2, pde.displacement)

        errs = []
        for _ in solver():
            errs.append(np.max(np.abs(pde.gfu_D.vec.FV().NumPy())))

        return np.max(np.array(errs))

    power_t = 1.5
    dt0 = 0.1
    dt_refs = 2
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h = 1.5
    dh0 = 0.2
    dh_refs = 2
    dhs = dh0/(power_h**(np.arange(dh_refs+1)))

    ERRORS_dX = np.zeros((len(dts), len(dhs)))
    ERRORS_dX.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):

            mesh, _ = generate_torus(maxh = dh, R = sqrt(2), r = 1)

            true_h = GridFunction(SurfaceL2(mesh, order = 0))
            true_h.Set(get_config().h, definedon = mesh.Boundaries('.*'))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            err = solve(mesh=mesh, dt=dt)

            ERRORS_dX[i, j] = err
    dhs = true_hs

    results_folder = './results_WillmoreBoundaryStabBDF1Model'
    os.makedirs(results_folder, exist_ok=True)

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_dX))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(results_folder + '/clifford_torus_stationary_th.dat', sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_dX.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(results_folder + '/clifford_torus_stationary_ht.dat', sep='\t', index=False)

def test_half_clifford_torus():

    Tend = 1
    def solve(mesh, dt):

        solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
        solvermesh = SolverMesh(mesh)
        solver = Solver(solvermesh, solvertime)

        input_params = {
            "clamped_bnd": 'bottom',
            "clamped_conormal": CF((0, 0, -1))
        }
        pde = WillmoreBoundaryStabBDF1Model(solver, 1, name = 'willmore_boundary_bdf1', input_params=input_params)
        displacement = DisplacementBoundaryBDF1Coupling(solver, 2, pde.displacement)

        errs = []
        for _ in solver():
            errs.append(np.max(np.abs(pde.gfu_D.vec.FV().NumPy())))

        return np.max(np.array(errs))

    power_t = 1.5
    dt0 = 0.1
    dt_refs = 2
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h = 1.5
    dh0 = 0.2
    dh_refs = 2
    dhs = dh0/(power_h**(np.arange(dh_refs+1)))

    ERRORS_dX = np.zeros((len(dts), len(dhs)))
    ERRORS_dX.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):

            mesh, _ = generate_half_torus(maxh = dh, R = sqrt(2), r = 1)

            true_h = GridFunction(SurfaceL2(mesh, order = 0))
            true_h.Set(get_config().h, definedon = mesh.Boundaries('.*'))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            err = solve(mesh=mesh, dt=dt)

            ERRORS_dX[i, j] = err
    dhs = true_hs

    results_folder = './results_WillmoreBoundaryStabBDF1Model'
    os.makedirs(results_folder, exist_ok=True)

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_dX))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(results_folder + '/half_clifford_torus_stationary_th.dat', sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_dX.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(results_folder + '/half_clifford_torus_stationary_ht.dat', sep='\t', index=False)

def test_sphere_spontaneous_mc():

    sp_curv = -1
    R0 = 1
    Tend = 1
    F = lambda t, s: -sp_curv/s*(2/s+sp_curv)
    t_eval = np.arange(0, Tend, Tend/1000)
    sol = solve_ivp(F, [0, Tend], [R0], t_eval=t_eval, dense_output=True)

    def solve(mesh, dt):

        gfu = GridFunction(H1(mesh, order = 1, definedon = mesh.Boundaries('.*')))

        solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
        solvermesh = SolverMesh(mesh)
        solver = Solver(solvermesh, solvertime)

        pde = WillmoreBoundaryStabBDF1Model(solver, 1, name = 'willmore_boundary_bdf1')
        pde.set_input_fields({
            "spontaneous_curvature": sp_curv, 
        })
        displacement = DisplacementBoundaryBDF1Coupling(solver, 2, pde.displacement)

        errs = []
        for _ in solver():

            Rt = sol.sol(solver.current_time+dt)
            gfu.Set(sqrt(x**2+y**2+z**2), dual = True, definedon = mesh.Boundaries('.*'))
            errs.append(np.max(np.abs(gfu.vec.FV().NumPy() - Rt)))

        return errs

    power_t = 2
    dt0 = 0.01
    dt_refs = 2
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h = 1.5
    dh0 = 0.2
    dh_refs = 2
    dhs = dh0/(power_h**(np.arange(dh_refs+1)))

    ERRORS_dX = np.zeros((len(dts), len(dhs)))
    ERRORS_dX.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):

            mesh, _ = generate_sphere(maxh = dh, R = R0)

            true_h = GridFunction(SurfaceL2(mesh, order = 0))
            true_h.Set(get_config().h, definedon = mesh.Boundaries('.*'))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            errs = solve(mesh=mesh, dt=dt)

            ERRORS_dX[i, j] = np.max(errs)
    dhs = true_hs

    results_folder = './results_WillmoreBoundaryStabBDF1Model'
    os.makedirs(results_folder, exist_ok=True)

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_dX))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(results_folder + '/sphere_spontaneous_mc_th.dat', sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_dX.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(results_folder + '/sphere_spontaneous_mc_ht.dat', sep='\t', index=False)

def test_torus21meshlab():
    
    Tend = 2
    dt = 1e-3
    n = 200

    solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
    mesh = Mesh('../data/torus.vol')
    solvermesh = SolverMesh(mesh)
    solver = Solver(solvermesh, solvertime)

    pde = WillmoreBoundaryStabBDF1Model(solver, 1, name = 'willmore_boundary_bdf1')
    displacement = DisplacementBoundaryBDF1Coupling(solver, 2, pde.displacement)

    energy = []
    time = []
    for _ in solver():
        energy_i = 0.5*Integrate(pde.gfu_k**2, mesh, BND)
        energy.append(energy_i)
        time.append(solver.current_time)

    results_folder = './results_WillmoreBoundaryStabBDF1Model'
    os.makedirs(results_folder, exist_ok=True)

    energy = np.array(energy)
    time = np.array(time)
    if len(time)>n:
        indices = np.linspace(0, len(time) - 1, n, dtype=int)
        energy = energy[indices]
        time = time[indices]

    df = pd.DataFrame(np.column_stack([time, energy]), columns = ['Time',  'mass'])
    df.to_csv(os.path.join(results_folder, 'energy_torus21meshlab_' + pde.name + '.dat'), sep='\t', index=False)

def test_torus21ngsolve():
    
    Tend = 2
    dt = 1e-3
    n = 200

    solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
    mesh, _ = generate_torus(maxh = 0.2, R = 2, r=1)
    solvermesh = SolverMesh(mesh)
    solver = Solver(solvermesh, solvertime)

    pde = WillmoreBoundaryStabBDF1Model(solver, 1, name = 'willmore_boundary_bdf1')
    displacement = DisplacementBoundaryBDF1Coupling(solver, 2, pde.displacement)

    energy = []
    time = []
    for _ in solver():
        energy_i = 0.5*Integrate(pde.gfu_k**2, mesh, BND)
        energy.append(energy_i)
        time.append(solver.current_time)

    results_folder = './results_WillmoreBoundaryStabBDF1Model'
    os.makedirs(results_folder, exist_ok=True)

    energy = np.array(energy)
    time = np.array(time)
    if len(time)>n:
        indices = np.linspace(0, len(time) - 1, n, dtype=int)
        energy = energy[indices]
        time = time[indices]

    df = pd.DataFrame(np.column_stack([time, energy]), columns = ['Time',  'mass'])
    df.to_csv(os.path.join(results_folder, 'energy_torus21ngsolve_' + pde.name + '.dat'), sep='\t', index=False)

def test_sigar511():
    
    Tend = 0.3
    dt = 1e-3
    n = 200

    solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
    mesh, _ = generate_sigar(maxh = 0.25, r=1, h=5)
    solvermesh = SolverMesh(mesh)
    solver = Solver(solvermesh, solvertime)

    pde = WillmoreBoundaryStabBDF1Model(solver, 1, name = 'willmore_boundary_bdf1')
    displacement = DisplacementBoundaryBDF1Coupling(solver, 2, pde.displacement)

    energy = []
    time = []
    for _ in solver():
        energy_i = 0.5*Integrate(pde.gfu_k**2, mesh, BND)
        energy.append(energy_i)
        time.append(solver.current_time)

    results_folder = './results_WillmoreBoundaryStabBDF1Model'
    os.makedirs(results_folder, exist_ok=True)

    energy = np.array(energy)
    time = np.array(time)
    if len(time)>n:
        indices = np.linspace(0, len(time) - 1, n, dtype=int)
        energy = energy[indices]
        time = time[indices]

    df = pd.DataFrame(np.column_stack([time, energy]), columns = ['Time',  'mass'])
    df.to_csv(os.path.join(results_folder, 'energy_sigar511_' + pde.name + '.dat'), sep='\t', index=False)

def test_sigar311():
    
    Tend = 1
    dt = 1e-3
    n = 200

    solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
    mesh, _ = generate_sigar(maxh = 0.25, r=1, h=3)
    solvermesh = SolverMesh(mesh)
    solver = Solver(solvermesh, solvertime)

    pde = WillmoreBoundaryStabBDF1Model(solver, 1, name = 'willmore_boundary_bdf1')
    displacement = DisplacementBoundaryBDF1Coupling(solver, 2, pde.displacement)

    energy = []
    time = []
    for _ in solver():
        energy_i = 0.5*Integrate(pde.gfu_k**2, mesh, BND)
        energy.append(energy_i)
        time.append(solver.current_time)

    results_folder = './results_WillmoreBoundaryStabBDF1Model'
    os.makedirs(results_folder, exist_ok=True)

    energy = np.array(energy)
    time = np.array(time)
    if len(time)>n:
        indices = np.linspace(0, len(time) - 1, n, dtype=int)
        energy = energy[indices]
        time = time[indices]

    df = pd.DataFrame(np.column_stack([time, energy]), columns = ['Time',  'mass'])
    df.to_csv(os.path.join(results_folder, 'energy_sigar311_' + pde.name + '.dat'), sep='\t', index=False)

tests = {
    "test_sphere": test_sphere,
    "test_half_sphere": test_half_sphere,
    "test_clifford_torus": test_half_sphere,
    "test_half_clifford_torus": test_half_clifford_torus,
    "test_sphere_spontaneous_mc": test_sphere_spontaneous_mc,
    "test_torus21meshlab": test_torus21meshlab,
    "test_torus21ngsolve": test_torus21ngsolve,
    "test_sigar511": test_sigar511,
    "test_sigar311": test_sigar311
}

with open("WillmoreBoundaryStabBDF1Model.txt", "w") as f:
    for name, test in tests.items():
        try:
            test()
        except Exception as e:
            f.write('Simulation ' + name + ' terminated with error \n ')
            f.write(str(e))
            f.write('\n')
