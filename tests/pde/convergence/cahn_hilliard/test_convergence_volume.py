from ngsolve import *
from cosmos import *
from ngsolve.webgui import Draw
from cosmos.utils.tools import gradient
import numpy as np
import logging
import pytest
import pandas as pd

logging.getLogger().setLevel(logging.INFO)

@pytest.fixture
def u0():
    return sin(x*y)**2

@pytest.fixture
def input_params(u0):
    params = {}
    params['u0'] = u0
    params['M'] = 0.01
    params['epsilon'] = 0.2
    params['sigma'] = 2
    params['fes_order'] = 1
    params['Neu_bnd_phase'] = '.*'
    params['Neu_bnd_potential'] = '.*'
    return params

def test_convergence_volume(
        request,
        artifacts_path,
        u0,
        input_params):
    
    out = artifacts_path
    filename = request.function.__name__

    Tend = 1
    def solve(mesh, dt):

        t = Parameter(0)
        u_ex = (u0*cos(t)**2).Compile()
        w_ex = sin(t).Compile()
        rhs_u = (u_ex.Diff(t) - input_params['M']*Trace(gradient(gradient(w_ex, Id(2)), Id(2)))).Compile()
        rhs_w = (w_ex - input_params['sigma']/input_params['epsilon']*((4*u_ex**3-6*u_ex**2+2*u_ex)/4) 
                 + input_params['sigma']*input_params['epsilon']*Trace(gradient(gradient(u_ex, Id(2)), Id(2)))).Compile()

        solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend, t_coef=t)
        solvermesh = SolverMesh(mesh)
        solver = Solver(solvermesh, solvertime, iter=False,
                        name = filename, printing=True)
        ch = CahnHilliardVolumeAlandBDF1Model(solver, 1, input_params=input_params)
        ch.set_input_fields({
            "rhs_u": rhs_u,
            "rhs_w": rhs_w
        })

        ch.set_input_fields({
            "grad_phase_bnd": (gradient(u_ex, Id(2))).Compile(),
            "grad_potential_bnd": (gradient(w_ex, Id(2))).Compile()
        })
        
        errs_phase = []
        errs_potential = []
        for _ in solver():
            err_u = sqrt(Integrate(InnerProduct(ch.phase-u_ex, ch.phase-u_ex), mesh, VOL_or_BND = VOL))
            errs_phase.append(err_u)
            err_w = sqrt(Integrate(InnerProduct(ch.potential-w_ex, ch.potential-w_ex), mesh, VOL_or_BND = VOL))
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

            mesh = Mesh(unit_square.GenerateMesh(maxh = dh))

            true_h = GridFunction(L2(mesh, order = 0))
            true_h.Set(get_config().h)
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
    df.to_csv(os.path.join(out, 'square_phase_th.dat'), sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_U.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out,'square_phase_ht.dat'), sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_W))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out,'square_potential_th.dat'), sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_W.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(os.path.join(out,'square_potential_ht.dat'), sep='\t', index=False)

    print(ERRORS_U)
    print(ERRORS_W)
    print(out)
    
    assert 1