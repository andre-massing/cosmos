# %%

from ngsolve import *
from cosmos import *
from cosmos.utils.generate_surface_meshes import generate_sphere
import os
import numpy as np
import pandas as pd
from myngspy import *
from scipy.integrate import solve_ivp

def test_1_willmore_boundary_v1_bdf1_model():

    sp_curv = -1
    R0 = 1
    Tend = 1
    F = lambda t, s: -sp_curv/s*(2/s+sp_curv)
    t_eval = np.arange(0, Tend, Tend/1000)
    sol = solve_ivp(F, [0, Tend], [R0], t_eval=t_eval, dense_output=True)

    def sphere_sp_curv(mesh, dt):

        gfu = GridFunction(H1(mesh, order = 1, definedon = mesh.Boundaries('.*')))

        solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
        solvermesh = SolverMesh(mesh)
        solver = Solver(solvermesh, solvertime)

        pde = WillmoreBoundaryV1BDF1Model(solver, 1, name = 'willmore_boundary_v1_bdf1')
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
    dt_refs = 3
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

            errs = sphere_sp_curv(mesh=mesh, dt=dt)

            ERRORS_dX[i, j] = np.max(errs)
    dhs = true_hs

    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_dX))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv('./results_test_1_willmore_boundary_v1_bdf1_model/data_dX.dat', sep='\t', index=False)

    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_dX.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv('./results_test_1_willmore_boundary_v1_bdf1_model/data_dX_flipped.dat', sep='\t', index=False)

test_1_willmore_boundary_v1_bdf1_model()