from ngsolve import *
from cosmos import *
from ngsolve.webgui import Draw
from cosmos.utils.generate_meshes import generate_boundary_sphere, generate_boundary_cylinder
from cosmos.utils.tools import gradient
import numpy as np
import logging
import pytest
import pandas as pd

logging.getLogger().setLevel(logging.INFO)

@pytest.fixture
def u0():
    return x*y

@pytest.fixture
def input_params(u0):
    params = {}
    params['u0'] = u0
    params['M'] = 0.01
    params['epsilon'] = 0.2
    params['sigma'] = 2
    params['fes_order'] = 2
    params['Neu_bnd_phase'] = 'bboundary'
    params['Neu_bnd_potential'] = 'bboundary'
    return params

@pytest.mark.parametrize("ch_solver", [
    CahnHilliardBoundaryBachiniBDF1Model,
    CahnHilliardBoundaryPolBDF1Model,
    CahnHilliardBoundaryElliottBDF1Model,
])
def test_convergence_moving(
        request,
        artifacts_path,
        u0,
        ch_solver,
        input_params):
    
    out = artifacts_path
    filename = request.function.__name__

    Tend = 1
    n_ex = Normalize(CF((x, y, 0)))
    P_ex = Id(3) - OuterProduct(n_ex, n_ex)

    def solve(mesh, dt):

        t = Parameter(0)
        def A_B(t):
            D = CF((1+0.5*sin(pi*t), 0, 0,\
                        0, 1+0.5*sin(pi*t), 0,\
                            0, 0, 1/(1+0.5*sin(pi*t))), dims = (3,3))
            # R = CF((cos(2*pi*t), -sin(2*pi*t), 0,\
            #             sin(2*pi*t), cos(2*pi*t), 0,\
            #                 0, 0, 1), dims = (3,3))
            # A = D*R
            A = D
            B = CF((0, 0, 0))
            return A, B
        A, B = A_B(t)
        detJ = Det(A)
        invA = Cof(A).trans/detJ
        inv_phi = invA*(CF((x,y,z)) - B) # Inverse of deformation map
        w_phi = A.Diff(t)*inv_phi + B.Diff(t)

        Ap, Bp = A_B(t+dt)
        displacement_ex = (Ap-A)*inv_phi+(Bp-B)

        u_ex = (u0*cos(t)).Compile()
        w_ex = sin(t).Compile()
        rhs_u = (u_ex.Diff(t) + w_phi*gradient(u_ex, Id(mesh.dim)) + Trace(gradient(w_phi, P_ex))*u_ex 
                 - input_params['M']*Trace(gradient(gradient(w_ex, P_ex), P_ex))).Compile()
        rhs_w = (w_ex - input_params['sigma']/input_params['epsilon']*(u_ex**3-u_ex) 
                 + input_params['sigma']*input_params['epsilon']*Trace(gradient(gradient(u_ex, P_ex), P_ex))).Compile()

        solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend, t_coef=t)
        solvermesh = SolverMesh(mesh)
        solver = Solver(solvermesh, solvertime, iter=False,
                        name = filename, printing=True)
        ch = ch_solver(solver, model_order = 2, input_params=input_params)
        ale = ALEModel(solver, model_order = 1)
        ale.set_bnd_displacement(displacement_ex, 'default')
        ch.set_input_fields({
            "rhs_u": rhs_u,
            "rhs_w": rhs_w
        })

        ch.set_input_fields({
            "grad_phase_bnd": (gradient(u_ex, P_ex)).Compile(),
            "grad_potential_bnd": (gradient(w_ex, P_ex)).Compile()
        })
        
        errs_phase = []
        errs_potential = []
        for _ in solver():
            err_u = sqrt(Integrate(InnerProduct(ch.phase-u_ex, ch.phase-u_ex), mesh, VOL_or_BND = BND))
            errs_phase.append(err_u)
            err_w = sqrt(Integrate(InnerProduct(ch.potential-w_ex, ch.potential-w_ex), mesh, VOL_or_BND = BND))
            errs_potential.append(err_w)

        ERR_U = np.sqrt(dt*np.sum(np.array(errs_phase)**2))
        ERR_W = np.sqrt(dt*np.sum(np.array(errs_potential)**2))

        return ERR_U, ERR_W

    power_t = 1.5
    dt0 = 0.02
    dt_refs = 3
    dts = dt0/(power_t**(np.arange(dt_refs+1)))

    power_h = 1.5
    dh0 = 0.2
    dh_refs = 3
    dhs = dh0/(power_h**(np.arange(dh_refs+1)))

    ERRORS_U = np.zeros((len(dts), len(dhs)))
    ERRORS_U.fill(np.inf)
    ERRORS_W = np.zeros((len(dts), len(dhs)))
    ERRORS_W.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):

            mesh = generate_boundary_cylinder(maxh = dh)

            true_h = GridFunction(SurfaceL2(mesh, order = 0))
            true_h.Set(get_config().h, definedon = mesh.Boundaries('.*'))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            err_u, err_w = solve(mesh=mesh, dt=dt)

            ERRORS_U[i, j] = err_u
            ERRORS_W[i, j] = err_w
    dhs = true_hs

    os.makedirs(out, exist_ok=True)

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_U))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out, 'sphere_phase_th.dat'), sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_U.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out,'sphere_phase_ht.dat'), sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_W))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out,'sphere_potential_th.dat'), sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_W.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out,'sphere_potential_ht.dat'), sep='\t', index=False)
    
    assert 1