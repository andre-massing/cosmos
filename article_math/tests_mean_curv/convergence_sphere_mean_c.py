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

testname = 'convergence_sphere_mean_c'

def test_sphere_mean_c(results_path, solver_name, pde_constructor, **init_kwargs):

    R0 = 1

    def sphere_mean_c(mesh, dt, folderpath, filename):

        gfu = GridFunction(H1(mesh, order = 1, definedon = mesh.Boundaries('.*')))

        t = Parameter(0.0)
        dt = Parameter(dt)
        T = 0.2

        mean_c = pde_constructor(name = [filename + 'dX', filename + 'mc'],
                        **init_kwargs)

        solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)
        solver.AddPDE(mean_c)
        solver.ale.deformation_field = mean_c.gfu.components[0]

        errs = []
        for _ in solver():

            Rt = sqrt(R0**2-4*t.Get())
            mesh.SetDeformation(solver.ale.deformation)
            gfu.Set(sqrt(x**2+y**2+z**2), dual = True, definedon = mesh.Boundaries('.*'))
            mesh.UnsetDeformation()
            errs.append(np.max(np.abs(gfu.vec.FV().NumPy() - Rt)))

        return errs


    folderpath = os.path.join(results_path, solver_name, testname)
    os.makedirs(folderpath, exist_ok=True)

    power_t = 1.5
    dt0 = 0.01
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

            errs = sphere_mean_c(mesh=mesh, dt=dt, folderpath=folderpath, filename=name)

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
