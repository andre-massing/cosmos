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

testname = 'convergence_sphere_scalar'

def test_sphere_scalar(results_path, solver_name, pde_constructor, **init_kwargs):

    def sphere(mesh, dt, folderpath, filename):

        t = Parameter(0.0)
        dt = Parameter(dt)
        T = 1

        H_ex = -2
        dX_ex = CF((0, 0, 0))

        willmore = pde_constructor(name = [filename + 'dX', filename + 'mc'],
                        **init_kwargs)

        err_save = SaveError(ex_sol = [dX_ex, H_ex], 
                    norm = ['L2norm', 'L2'], 
                    folderpath = folderpath,
                    filename = filename)
        willmore.SaveErr(err_save)

        solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)
        solver.AddPDE(willmore)
        solver.ale.deformation_field = willmore.gfu.components[0]

        solver.Solve()

    folderpath = os.path.join(results_path, solver_name, testname)

    power_t = 1.5
    dt0 = 0.1
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
    ERRORS_mc = np.zeros((len(dts), len(dhs)))
    ERRORS_mc.fill(np.inf)

    true_hs = np.zeros(len(dhs))
    for i, dt in enumerate(dts):
        for j, dh in enumerate(dhs):

            name = 'error' + str(i) + str(j)
            mesh, _ = generate_sphere(maxh = dh, R = 1)

            true_h = GridFunction(SurfaceL2(mesh, order = 0))
            true_h.Set(MyMeshSize(), definedon = mesh.Boundaries('.*'))
            true_h = np.max(true_h.vec.data)
            true_hs[j] = true_h

            sphere(mesh=mesh, dt=dt, folderpath=folderpath, filename=name)

            file_path = os.path.join(folderpath, name)
            df = pd.read_csv(file_path)
            errs = df[name + 'dX']
            ERRORS_dX[i, j] = np.sqrt(np.sum(dt*errs**2))

            file_path = os.path.join(folderpath, name)
            df = pd.read_csv(file_path)
            errs = df[name + 'mc']
            ERRORS_mc[i, j] = np.sqrt(np.sum(dt*errs**2))
    dhs = true_hs

    print('Sneak look at overall convergence displacement')
    print(np.log(ERRORS_dX.diagonal()[:-1]/ERRORS_dX.diagonal()[1:])/np.log(np.maximum(power_h, power_t)))
    print('Sneak look at overall convergence mean curvature')
    print(np.log(ERRORS_mc.diagonal()[:-1]/ERRORS_mc.diagonal()[1:])/np.log(np.maximum(power_h, power_t)))

    dict_dX = {
        'dhs': dhs,
        'power_h': power_h,
        'dts': dts,
        'power_t': power_t,
        'errors': ERRORS_dX
    }

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

    dict_mc = {
        'dhs': dhs,
        'power_h': power_h,
        'dts': dts,
        'power_t': power_t,
        'errors': ERRORS_mc
    }

    name = os.path.join(folderpath, 'data_mc.dat')
    labels =  [f'{x:.2e}' for x in dhs]
    labels = ["th"] + labels
    output = np.column_stack((dts, ERRORS_mc))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(name, sep='\t', index=False)

    name = os.path.join(folderpath, 'data_mc_flipped.dat')
    labels =  [f'{x:.2e}' for x in dts]
    labels = ["ht"] + labels
    output = np.column_stack((dhs, ERRORS_mc.transpose()))
    df = pd.DataFrame(output, columns=labels)
    df.to_csv(name, sep='\t', index=False)
