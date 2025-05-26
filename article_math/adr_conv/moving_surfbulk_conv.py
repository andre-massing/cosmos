# %%

import numpy as np
from cosmos.utils.generate_surface_meshes import generate_circle
from moving_surfbulk import solve_moving_surfbulk_adr
import pandas as pd
import os

power_t = 1.5
dt0 = 0.02
dt_refs = 4
dts = dt0/(power_t**(np.arange(dt_refs+1)))
print('Convergence time-steps:', dts)

power_h = 1.5
dh0 = 0.1
dh_refs = 4
dhs = dh0/(power_h**(np.arange(dh_refs+1)))
print('Convergence mesh-sizes:', dhs)

folderpath = './moving_surfbulk_adr'
ERRORS_B = np.zeros((len(dts), len(dhs)))
ERRORS_B.fill(np.inf)
ERRORS_S = np.zeros((len(dts), len(dhs)))
ERRORS_S.fill(np.inf)

for i, dt in enumerate(dts):
    for j, dh in enumerate(dhs):

        name = 'error' + str(i) + str(j)
        mesh, _ = generate_circle(maxh = dh)

        solve_moving_surfbulk_adr(mesh=mesh, dt=dt, folderpath=folderpath, filename=name)

        file_path = os.path.join(folderpath, name + 'bulk')
        df = pd.read_csv(file_path)
        errs = df[name + 'bulk']
        ERRORS_B[i, j] = np.sqrt(np.sum(dt*errs**2))

        file_path = os.path.join(folderpath, name + 'surf')
        df = pd.read_csv(file_path)
        errs = df[name + 'surf']
        ERRORS_S[i, j] = np.sqrt(np.sum(dt*errs**2))

dict = {
    'dhs': dhs,
    'power_h': power_h,
    'dts': dts,
    'power_t': power_t,
    'errors': ERRORS_B
}
np.save('moving_surfbulk_bulk',dict)

name = os.path.join(folderpath, 'data_bulk.dat')
labels =  [f'{x:.2e}' for x in dhs]
labels = ["th"] + labels
output = np.column_stack((dts, ERRORS_B))
df = pd.DataFrame(output, columns=labels)
df.to_csv(name, sep='\t', index=False)

name = os.path.join(folderpath, 'data_bulk_flipped.dat')
labels =  [f'{x:.2e}' for x in dts]
labels = ["ht"] + labels
output = np.column_stack((dhs, ERRORS_B.transpose()))
df = pd.DataFrame(output, columns=labels)
df.to_csv(name, sep='\t', index=False)

dict = {
    'dhs': dhs,
    'power_h': power_h,
    'dts': dts,
    'power_t': power_t,
    'errors': ERRORS_S
}
np.save('moving_surfbulk_surf',dict)

name = os.path.join(folderpath, 'data_surf.dat')
labels =  [f'{x:.2e}' for x in dhs]
labels = ["th"] + labels
output = np.column_stack((dts, ERRORS_S))
df = pd.DataFrame(output, columns=labels)
df.to_csv(name, sep='\t', index=False)

name = os.path.join(folderpath, 'data_surf_flipped.dat')
labels =  [f'{x:.2e}' for x in dts]
labels = ["ht"] + labels
output = np.column_stack((dhs, ERRORS_S.transpose()))
df = pd.DataFrame(output, columns=labels)
df.to_csv(name, sep='\t', index=False)