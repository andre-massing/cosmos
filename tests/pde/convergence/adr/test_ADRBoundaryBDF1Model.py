from ngsolve import *
from cosmos import *
from cosmos.pdes.pde_tools import gradient
from cosmos.utils.generate_surface_meshes import generate_half_sphere, generate_sphere
import os
import numpy as np
import pandas as pd
from ngsolve.webgui import Draw
from myngspy import *

def test_half_sphere_dir_bnd():

    def solve(mesh, dt):

        t = Parameter(0.0)
        T = 1
        # Running Simulation
        solvertime = SolverTime(dt = dt, initial_t=0, final_t=T, t_coef=t)
        solvermesh = SolverMesh(mesh)
        solver = Solver(solvermesh, solvertime)

        def A_B(t):
            A = CF((cos(t), -sin(t), 0,\
                        sin(t), cos(t), 0,\
                            0, 0, 1), dims = (3,3))
            B = CF((0, 0, 0))
            return A, B
        A, B = A_B(t)
        detJ = Det(A)
        invA = Cof(A).trans/detJ
        inv_phi = invA*(CF((x,y,z)) - B) # Inverse of deformation map
        w_phi = A.Diff(t)*inv_phi + B.Diff(t) # velocity of the moving domain
        n_ex = Cof(A)*inv_phi/Norm(Cof(A)*inv_phi)
        P_ex = Id(3) - OuterProduct(n_ex, n_ex)

        deformation_ex =  A*inv_phi+B - inv_phi
        deformation = DeformationBoundaryBDF1Coupling(solver, 1, deformation_ex, name='deformation')

        # Ap, Bp = A_B(t-dt)
        # displacement_ex = (A-Ap)*inv_phi+(B-Bp)
        # displacement = DisplacementBoundaryBDF1Coupling(solver, 1, displacement_ex, name='deformation')

        u0 = cos(2*pi*x)
        u_ex = u0*cos(t) # Exact solution
        c = 1 + t**2 # reaction coefficient
        b = P_ex*CF((2,-x, 1))
        flux_b = b*u_ex
        rel_flux = u_ex*Trace(gradient(w_phi, P_ex)) + w_phi*gradient(u_ex, Id(3))
        rhs = u_ex.Diff(t) + rel_flux + Trace(gradient(flux_b, P_ex)) + c*u_ex # manufactured solution right-hand side

        input_params = {
            "u0": u0,
            "Dir_bnd": 'bottom',
            "fes_order": 1
        }
        pde = ADRBoundaryBDF1Model(solver, 2, name = 'adr_boundary_bdf1', input_params = input_params)
        pde.set_input_fields({
            "b": b,
            "c": c,
            "u_bnd": u_ex,
            "gradu_bnd": gradient(u_ex, P_ex),
            "rhs": rhs
        })

        errs = []
        for _ in solver():
            err_i = Integrate(InnerProduct(pde.sol - u_ex, pde.sol - u_ex), mesh, BND)
            errs.append(err_i)

        return np.sqrt(dt*np.sum(np.array(errs)**2))

    power_t = 1.5
    dt0 = 0.02
    dt_refs = 0
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h = 1.5
    dh0 = 0.1
    dh_refs = 0
    dhs = dh0/(power_h**(np.arange(dh_refs+1)))

    ERRORS = np.zeros((len(dts), len(dhs)))
    ERRORS.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):

            name = 'error' + str(i) + str(j)
            mesh, _ = generate_half_sphere(maxh = dh)

            true_h = GridFunction(SurfaceL2(mesh, order = 0))
            true_h.Set(MyMeshSize(), definedon = mesh.Boundaries('.*'))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            err = solve(mesh=mesh, dt=dt)
            ERRORS[i, j] = err

    dhs = true_hs

    os.makedirs('./results_ADRBoundaryBDF1Model', exist_ok=True)

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv('./results_ADRBoundaryBDF1Model/pure_convection_sphere_dir_bnd_th', sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv('./results_ADRBoundaryBDF1Model/pure_convection_sphere_dir_bnd_ht', sep='\t', index=False)

def test_half_sphere_neu_bnd():

    def solve(mesh, dt):

        t = Parameter(0.0)
        T = 1
        # Running Simulation
        solvertime = SolverTime(dt = dt, initial_t=0, final_t=T, t_coef=t)
        solvermesh = SolverMesh(mesh)
        solver = Solver(solvermesh, solvertime)

        def A_B(t):
            A = CF((cos(t), -sin(t), 0,\
                        sin(t), cos(t), 0,\
                            0, 0, 1), dims = (3,3))
            B = CF((0, 0, 0))
            return A, B
        A, B = A_B(t)
        detJ = Det(A)
        invA = Cof(A).trans/detJ
        inv_phi = invA*(CF((x,y,z)) - B) # Inverse of deformation map
        w_phi = A.Diff(t)*inv_phi + B.Diff(t) # velocity of the moving domain
        n_ex = Cof(A)*inv_phi/Norm(Cof(A)*inv_phi)
        P_ex = Id(3) - OuterProduct(n_ex, n_ex)

        deformation_ex =  A*inv_phi+B - inv_phi
        deformation = DeformationBoundaryBDF1Coupling(solver, 1, deformation_ex, name='deformation')

        # Ap, Bp = A_B(t-dt)
        # displacement_ex = (A-Ap)*inv_phi+(B-Bp)
        # displacement = DisplacementBoundaryBDF1Coupling(solver, 1, displacement_ex, name='deformation')

        u0 = cos(2*pi*x)
        u_ex = u0*cos(t) # Exact solution
        c = 1 + t**2 # reaction coefficient
        b = P_ex*CF((2,-x, 1))
        flux_b = b*u_ex
        rel_flux = u_ex*Trace(gradient(w_phi, P_ex)) + w_phi*gradient(u_ex, Id(3))
        rhs = u_ex.Diff(t) + rel_flux + Trace(gradient(flux_b, P_ex)) + c*u_ex # manufactured solution right-hand side

        input_params = {
            "u0": u0,
            "Neu_bnd": 'bottom',
            "fes_order": 1
        }
        pde = ADRBoundaryBDF1Model(solver, 2, name = 'adr_boundary_bdf1', input_params = input_params)
        pde.set_input_fields({
            "b": b,
            "c": c,
            "u_bnd": u_ex,
            "gradu_bnd": gradient(u_ex, P_ex),
            "rhs": rhs
        })

        errs = []
        for _ in solver():
            err_i = Integrate(InnerProduct(pde.sol - u_ex, pde.sol - u_ex), mesh, BND)
            errs.append(err_i)

        return np.sqrt(dt*np.sum(np.array(errs)**2))

    power_t = 1.5
    dt0 = 0.02
    dt_refs = 0
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h = 1.5
    dh0 = 0.1
    dh_refs = 0
    dhs = dh0/(power_h**(np.arange(dh_refs+1)))

    ERRORS = np.zeros((len(dts), len(dhs)))
    ERRORS.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):

            name = 'error' + str(i) + str(j)
            mesh, _ = generate_half_sphere(maxh = dh)

            true_h = GridFunction(SurfaceL2(mesh, order = 0))
            true_h.Set(MyMeshSize(), definedon = mesh.Boundaries('.*'))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            err = solve(mesh=mesh, dt=dt)
            ERRORS[i, j] = err

    dhs = true_hs

    os.makedirs('./results_ADRBoundaryBDF1Model', exist_ok=True)

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv('./results_ADRBoundaryBDF1Model/pure_convection_sphere_neu_bnd_th', sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv('./results_ADRBoundaryBDF1Model/pure_convection_sphere_neu_bnd_ht', sep='\t', index=False)

def test_sphere():

    def solve(mesh, dt):

        t = Parameter(0.0)
        T = 1
        # Running Simulation
        solvertime = SolverTime(dt = dt, initial_t=0, final_t=T, t_coef=t)
        solvermesh = SolverMesh(mesh)
        solver = Solver(solvermesh, solvertime)

        def A_B(t):
            A = CF((cos(t), -sin(t), 0,\
                        sin(t), cos(t), 0,\
                            0, 0, 1), dims = (3,3))
            B = CF((0, 0, 0))
            return A, B
        A, B = A_B(t)
        detJ = Det(A)
        invA = Cof(A).trans/detJ
        inv_phi = invA*(CF((x,y,z)) - B) # Inverse of deformation map
        w_phi = A.Diff(t)*inv_phi + B.Diff(t) # velocity of the moving domain
        n_ex = Cof(A)*inv_phi/Norm(Cof(A)*inv_phi)
        P_ex = Id(3) - OuterProduct(n_ex, n_ex)

        deformation_ex =  A*inv_phi+B - inv_phi
        deformation = DeformationBoundaryBDF1Coupling(solver, 1, deformation_ex, name='deformation')

        # Ap, Bp = A_B(t-dt)
        # displacement_ex = (A-Ap)*inv_phi+(B-Bp)
        # displacement = DisplacementBoundaryBDF1Coupling(solver, 1, displacement_ex, name='deformation')

        u0 = cos(2*pi*x)
        u_ex = u0*cos(t) # Exact solution
        c = 1 + t**2 # reaction coefficient
        b = P_ex*CF((2,-x, 1))
        flux_b = b*u_ex
        rel_flux = u_ex*Trace(gradient(w_phi, P_ex)) + w_phi*gradient(u_ex, Id(3))
        rhs = u_ex.Diff(t) + rel_flux + Trace(gradient(flux_b, P_ex)) + c*u_ex # manufactured solution right-hand side

        input_params = {
            "u0": u0,
            "fes_order": 1
        }
        pde = ADRBoundaryBDF1Model(solver, 2, name = 'adr_boundary_bdf1', input_params = input_params)
        pde.set_input_fields({
            "b": b,
            "c": c,
            "rhs": rhs
        })

        errs = []
        for _ in solver():
            err_i = Integrate(InnerProduct(pde.sol - u_ex, pde.sol - u_ex), mesh, BND)
            errs.append(err_i)

        return np.sqrt(dt*np.sum(np.array(errs)**2))

    power_t = 1.5
    dt0 = 0.02
    dt_refs = 0
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h = 1.5
    dh0 = 0.1
    dh_refs = 0
    dhs = dh0/(power_h**(np.arange(dh_refs+1)))

    ERRORS = np.zeros((len(dts), len(dhs)))
    ERRORS.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):

            name = 'error' + str(i) + str(j)
            mesh, _ = generate_half_sphere(maxh = dh)

            true_h = GridFunction(SurfaceL2(mesh, order = 0))
            true_h.Set(MyMeshSize(), definedon = mesh.Boundaries('.*'))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            err = solve(mesh=mesh, dt=dt)
            ERRORS[i, j] = err

    dhs = true_hs

    os.makedirs('./results_ADRBoundaryBDF1Model', exist_ok=True)

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv('./results_ADRBoundaryBDF1Model/pure_convection_sphere_th', sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv('./results_ADRBoundaryBDF1Model/pure_convection_sphere_ht', sep='\t', index=False)

def test_boundary_layer():

    dt = 0.01
    T = 1
    n = 200
    mesh, _ = generate_half_sphere(maxh = 0.1)
    sample_rate = np.maximum(int(T/dt/n), 1)

    dt = Parameter(0.01)
    t = Parameter(0.0)
    solvertime = SolverTime(dt = dt, initial_t=0, final_t=T, t_coef=t)
    solvermesh = SolverMesh(mesh)
    solver = Solver(solvermesh, solvertime, name = 'ADRBoundaryBDF1Model_illposed_case')

    mass = []
    time = []

    def AandB(t):
        A = CF((cos(t), -sin(t), 0,\
                        sin(t), cos(t), 0,\
                            0, 0, 1), dims = (3,3))
        B = CF((0, 0, 0))
        return A, B

    A, B = AandB(t)
    detJ = Det(A)
    invA = Cof(A).trans/detJ
    inv_phi = invA*(CF((x,y,z)) - B) # Inverse of deformation map
    w_phi = A.Diff(t)*inv_phi + B.Diff(t)
    n_ex = Cof(A)*inv_phi/Norm(Cof(A)*inv_phi)
    P_ex = Id(3) - OuterProduct(n_ex, n_ex)
    A1, B1 = AandB(t+dt)
    dX_ex = (A1-A)*inv_phi+(B1-B)

    deformation_ex =  A*inv_phi+B - inv_phi
    deformation = DeformationBoundaryBDF1Coupling(solver, 1, deformation_ex, name='deformation')

    # Ap, Bp = A_B(t-dt)
    # displacement_ex = (A-Ap)*inv_phi+(B-Bp)
    # displacement = DisplacementBoundaryBDF1Coupling(solver, 1, displacement_ex, name='deformation')


    u0 = exp(-3*(x**2 + y**2))
    b = CF((z,0,-x))*(1-exp(-10*z))

    input_params = {
        "u0": u0,
        "Neu_bnd": 'bottom',
        "fes_order": 1
    }
    pde = ADRBoundaryBDF1Model(solver, 2, name = 'adr_boundary_bdf1', input_params = input_params)
    pde.set_input_fields({
        "b": b,
        "gradu_bnd": CF((0, 0, 0))
    })

    solver.save_model_solution('adr_boundary_bdf1', 'sol', sample_rate=sample_rate)

    for sol in solver():
        mass_i = Integrate(pde.sol, mesh, BND)
        mass.append(mass_i)
        time.append(t.Get())

    mass = np.array(mass)
    mass = (mass - mass[0])/mass[0]

    mass = np.array(mass)
    time = np.array(time)
    if len(time)>n:
        indices = np.linspace(0, len(time) - 1, n, dtype=int)
        mass = mass[indices]
        time = time[indices]

    df = pd.DataFrame(np.column_stack([time, mass]), columns = ['Time',  'mass'])
    os.makedirs('./results_ADRBoundaryBDF1Model', exist_ok=True)
    df.to_csv(os.path.join('./results_ADRBoundaryBDF1Model', 'mass_' + pde.name + '.dat'), sep='\t', index=False)

test_half_sphere_dir_bnd()
test_half_sphere_neu_bnd()
test_sphere()
test_boundary_layer()