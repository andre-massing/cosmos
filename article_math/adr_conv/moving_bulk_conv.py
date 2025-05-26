# %%

import numpy as np
from cosmos.utils.generate_surface_meshes import generate_circle
from moving_bulk import solve_moving_bulk_adr
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

folderpath = './moving_bulk_adr'
ERRORS = np.zeros((len(dts), len(dhs)))
ERRORS.fill(np.inf)

for i, dt in enumerate(dts):
    for j, dh in enumerate(dhs):

        name = 'error' + str(i) + str(j)
        mesh, _ = generate_circle(maxh = dh)

        solve_moving_bulk_adr(mesh=mesh, dt=dt, folderpath=folderpath, filename=name)

        file_path = os.path.join(folderpath, name)
        df = pd.read_csv(file_path)
        errs = df[name]

        ERRORS[i, j] = np.sqrt(np.sum(dt*errs**2))

dict = {
    'dhs': dhs,
    'power_h': power_h,
    'dts': dts,
    'power_t': power_t,
    'errors': ERRORS
}

np.save('moving_bulk',dict)

name = os.path.join(folderpath, 'data.dat')
labels =  [f'{x:.2e}' for x in dhs]
labels = ["th"] + labels
output = np.column_stack((dts, ERRORS))
df = pd.DataFrame(output, columns=labels)
df.to_csv(name, sep='\t', index=False)

name = os.path.join(folderpath, 'data_flipped.dat')
labels =  [f'{x:.2e}' for x in dts]
labels = ["ht"] + labels
output = np.column_stack((dhs, ERRORS.transpose()))
df = pd.DataFrame(output, columns=labels)
df.to_csv(name, sep='\t', index=False)