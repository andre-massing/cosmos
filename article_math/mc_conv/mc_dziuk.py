# %%
from cosmos.solvers.solvers import Dynamic
from cosmos.solvers.time_schemes import BDF1, BDF2
from cosmos.pdes.pde_mc_dziuk import MCDziuk
from cosmos.utils.generate_surface_meshes import generate_sphere
from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np
import pandas as pd

foldername = 'mc_dziuk'
folderpath = './' + foldername

def mc_dziuk(mesh, dt, folderpath, filename):

    t = Parameter(0.0)
    dt = Parameter(dt)
    T = 0.1

    R = 1
    R_ex = sqrt(R**2 - 4*t)
    n_ex = CF((x, y, z))/Norm(CF((x, y, z)))
    dX_ex = n_ex*(R_ex - R)
    H_ex = -2/R_ex*n_ex

    # With boundary conditions
    mc = MCDziuk(name = filename + 'dX', time_scheme=BDF1())

    from cosmos.pdes.pde_tools import SaveError
    err_save = SaveError(ex_sol = dX_ex, 
                    norm = 'L2norm', 
                    folderpath = folderpath,
                    filename = filename)
    mc.SaveErr(err_save)

    solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)
    solver.AddPDE(mc)
    solver.ale.deformation_field = mc.gfu

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
        mesh, _ = generate_sphere(maxh = dh, R = 1)

        mc_dziuk(mesh=mesh, dt=dt, folderpath=folderpath, filename=name)

        file_path = os.path.join(folderpath, name)
        df = pd.read_csv(file_path)
        errs = df[name + 'dX']
        ERRORS_dX[i, j] = np.sqrt(np.sum(dt*errs**2))


print('Sneak look at overall convergence displacement')
print(np.log(ERRORS_dX.diagonal()[:-1]/ERRORS_dX.diagonal()[1:])/np.log(np.maximum(power_h, power_t)))

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

import pandas as pd
import os

name = 'error00'
file_path = os.path.join(folderpath, name)

# Read the CSV into a DataFrame
df = pd.read_csv(file_path)

import matplotlib.pyplot as plt
plt.plot(df['Time'], df[name+'dX'], label = 'dX')
plt.xlabel('Time')
plt.ylabel('L2 Norm')
plt.grid()
plt.legend()
plt.show
# %%
