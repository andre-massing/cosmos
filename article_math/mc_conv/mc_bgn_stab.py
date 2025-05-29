# %%
from cosmos.solvers.solver_unsteady import UnsteadySolver
from cosmos.solvers.schemes import BDF1, BDF2
from cosmos.pdes.pde_mc_bgn_stab import MCBGNStab
from cosmos.utils.generate_surface_meshes import generate_sphere
from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np
import pandas as pd

foldername = 'mc_bgn_stab'
folderpath = './' + foldername

def mc_bgn_stab(mesh, dt, folderpath, filename):

    t = Parameter(0.0)
    dt = Parameter(dt)
    T = 0.1

    R = 1
    R_ex = sqrt(R**2 - 4*t)
    n_ex = CF((x, y, z))/Norm(CF((x, y, z)))
    dX_ex = n_ex*(R_ex - R)
    H_ex = -2/R_ex*n_ex

    # With boundary conditions
    mc = MCBGNStab(name = [filename + 'dX', filename + 'mc'], 
                               postprocess = False, stab = 1e-3, mc0 = H_ex)

    from cosmos.pdes.pde_tools import SaveError
    err_save = SaveError(ex_sol = [dX_ex, H_ex], 
                    norm = 'L2norm', 
                    folderpath = folderpath,
                    filename = filename)
    mc.SaveErr(err_save)

    solver = UnsteadySolver(mesh = mesh, dt=dt, T=T, t=t)
    solver.AddPDE(mc, BDF1(conservative=True))
    solver.ale.SetMeshDeformation(mc.gfu.components[0])

    solver.Solve()

power_t = 1.5
dt0 = 0.005
dt_refs = 3
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

for i, dt in enumerate(dts):
    for j, dh in enumerate(dhs):

        name = 'error' + str(i) + str(j)
        mesh, _ = generate_sphere(maxh = dh, R=1)

        mc_bgn_stab(mesh=mesh, dt=dt, folderpath=folderpath, filename=name)

        file_path = os.path.join(folderpath, name)
        df = pd.read_csv(file_path)
        errs = df[name + 'dX']
        ERRORS_dX[i, j] = np.sqrt(np.sum(dt*errs**2))

        file_path = os.path.join(folderpath, name)
        df = pd.read_csv(file_path)
        errs = df[name + 'mc']
        ERRORS_mc[i, j] = np.sqrt(np.sum(dt*errs**2))

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

import pandas as pd
import os

name = 'error00'
file_path = os.path.join(folderpath, name)

# Read the CSV into a DataFrame
df = pd.read_csv(file_path)

import matplotlib.pyplot as plt
plt.plot(df['Time'], df[name+'dX'], label = 'dX')
plt.plot(df['Time'], df[name+'mc'], label = 'mc')
plt.xlabel('Time')
plt.ylabel('L2 Norm')
plt.grid()
plt.legend()
plt.show
# %%
