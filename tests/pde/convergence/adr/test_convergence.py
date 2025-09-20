from ngsolve import *
from cosmos import *
from ngsolve.webgui import Draw
from cosmos.utils.generate_meshes import generate_boundary_sphere, generate_boundary_half_sphere
from cosmos.utils.tools import gradient
import numpy as np
import logging
import pytest
import pandas as pd

logging.getLogger().setLevel(logging.INFO)

@pytest.fixture
def u0():
    return cos(2*pi*x)

@pytest.fixture
def input_params(params_type, u0, bnd):
    if params_type == 0:
        params = {
            "u0": u0,
        }
    elif params_type == 1:
        params = {
            "u0": u0,
            "mass_preserving": True,
        }
    elif params_type == 2:
        params = {
            "u0": u0,
            "bounds": [-1, 1],
        }
    elif params_type == 3:
        params = {
            "u0": u0,
            "mass_preserving": True,
            "bounds": [-1, 1],
        }
    if bnd == 'neu':
        params['Neu_bnd'] = 'bboundary'
    elif bnd == 'dir':
        params['Dir_bnd'] = 'bboundary'
    return params

@pytest.mark.parametrize("adr_solver", [
    ADRBoundaryBDF1Model,
    ADRBoundaryBDF2Model,
    ADRBoundaryStabBDF1Model,
    ADRBoundaryStabBDF2Model
])
@pytest.mark.parametrize("params_type", [0, 1, 2, 3])
@pytest.mark.parametrize("bnd", [None, 'dir', 'neu'])
def test_convergence(
        request,
        artifacts_path,
        u0,
        adr_solver,
        input_params,
        bnd):
    
    out = artifacts_path
    filename = request.function.__name__

    def solve(mesh, dt):

        t = Parameter(0.0)
        T = 1
        # Running Simulation
        solvertime = SolverTime(dt = dt, initial_t=0, final_t=T, t_coef=t)
        solvermesh = SolverMesh(mesh)
        solver = Solver(solvermesh, solvertime, printing=True)

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

        Ap, Bp = A_B(t+dt)
        displacement_ex = (Ap-A)*inv_phi+(Bp-B)
        ale = ALEModel(solver=solver, model_order=1, name='displacement')
        ale.set_bnd_displacement(displacement_ex, 'default')

        u_ex = u0*cos(t) # Exact solution
        c = 1 + t**2 # reaction coefficient
        b = (P_ex*CF((2,-x, 1)))
        flux_b = b*u_ex
        rel_flux = u_ex*Trace(gradient(w_phi, P_ex)) + w_phi*gradient(u_ex, Id(3))
        rhs = (u_ex.Diff(t) + rel_flux + Trace(gradient(flux_b, P_ex)) + c*u_ex) # manufactured solution right-hand side

        pde = adr_solver(solver, 2, input_params = input_params)
        pde.set_input_fields({
            "b": b,
            "c": c,
            "rhs": rhs
        })
        if bnd == 'neu':
            pde.set_input_fields({
                "u_bnd": u_ex,
                "gradu_bnd": gradient(u_ex, P_ex),
            })
        elif bnd == 'dir':
            pde.set_input_fields({
                "u_bnd": u_ex,
            })

        errs = []
        for _ in solver():
            err_i = Integrate(InnerProduct(pde.sol - u_ex, pde.sol - u_ex), mesh, BND)
            errs.append(err_i)

        return np.sqrt(dt*np.sum(np.array(errs)))

    power_t = 1.5
    dt0 = 0.01
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

            if bnd == 'dir' or bnd == 'neu':
                mesh = generate_boundary_half_sphere(maxh = dh, R = 1)
            else:
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
    df.to_csv(os.path.join(out, 'adr_convergence_th.dat'), sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_dX.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out, 'adr_convergence_ht.dat'), sep='\t', index=False)
    
    assert 1