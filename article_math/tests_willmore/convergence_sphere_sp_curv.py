from ngsolve import *
from cosmos.pdes.pde_tools import gradient
from cosmos.utils.generate_surface_meshes import generate_sphere
from cosmos.pdes.pde_tools import SaveError
import os
import numpy as np
import pandas as pd
from cosmos.solvers.solvers import Dynamic
from ngsolve.webgui import Draw
from myngspy import *
from scipy.integrate import solve_ivp

testname = 'convergence_sphere_sp_curv'

def test_sphere_sp_curv(results_path, solver_name, pde_constructor, **init_kwargs):

    sp_curv = -1
    R0 = 1
    Tend = 1
    F = lambda t, s: -sp_curv/s*(2/s+sp_curv)
    t_eval = np.arange(0, Tend, Tend/1000)
    sol = solve_ivp(F, [0, Tend], [R0], t_eval=t_eval, dense_output=True)

    def sphere_sp_curv(mesh, dt, folderpath, filename):

        gfu = GridFunction(H1(mesh, order = 1, definedon = mesh.Boundaries('.*')))

        t = Parameter(0.0)
        dt = Parameter(dt)
        T = Tend

        willmore = pde_constructor(name = [filename + 'dX', filename + 'mc'], sp_curv = sp_curv,
                        **init_kwargs)

        solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)
        solver.AddPDE(willmore)
        solver.ale.deformation_field = willmore.gfu.components[0]

        errs = []
        for _ in solver():

            Rt = sol.sol(t.Get())
            mesh.SetDeformation(solver.ale.deformation)
            gfu.Set(sqrt(x**2+y**2+z**2), dual = True, definedon = mesh.Boundaries('.*'))
            mesh.UnsetDeformation()
            errs.append(np.max(np.abs(gfu.vec.FV().NumPy() - Rt)))

        return errs


    folderpath = os.path.join(results_path, solver_name, testname)
    os.makedirs(folderpath, exist_ok=True)

    power_t = 1.5
    dt0 = 0.02
    dt_refs = 2
    dts = dt0/(power_t**(np.arange(dt_refs+1)))
    print('Convergence time-steps:', dts)

    power_h = 1.5
    dh0 = 0.2
    dh_refs = 2
    dhs = dh0/(power_h**(np.arange(dh_refs+1)))
    print('Convergence mesh-sizes:', dhs)

    ERRORS_dX = np.zeros((len(dts), len(dhs)))
    ERRORS_dX.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):

            name = 'error' + str(i) + str(j)
            mesh, _ = generate_sphere(maxh = dh, R = R0)

            true_h = GridFunction(SurfaceL2(mesh, order = 0))
            true_h.Set(MyMeshSize(), definedon = mesh.Boundaries('.*'))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            errs = sphere_sp_curv(mesh=mesh, dt=dt, folderpath=folderpath, filename=name)

            ERRORS_dX[i, j] = np.max(errs)
    dhs = true_hs

    print('Sneak look at overall convergence displacement')
    print(np.log(ERRORS_dX.diagonal()[:-1]/ERRORS_dX.diagonal()[1:])/np.log(np.maximum(power_h, power_t)))


    name = os.path.join(folderpath, 'data_dX.dat')
    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_dX))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(name, sep='\t', index=False)

    name = os.path.join(folderpath, 'data_dX_flipped.dat')
    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_dX.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(name, sep='\t', index=False)
