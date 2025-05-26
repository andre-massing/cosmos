# %%

import numpy as np
from cosmos.utils.generate_surface_meshes import generate_half_sphere
from epsilon_adr import solve_epsilon_adr
import pandas as pd
import os
import matplotlib.pyplot as plt

dt = 0.01
mesh, _ = generate_half_sphere(maxh = 0.1)

options = [0, 1, 2]
folderpath = './epsilon'

for opt in options:
    name = 'epsilon'
    mass = solve_epsilon_adr(mesh=mesh, dt=dt, folderpath=folderpath, filename=name, option=opt)

    name = os.path.join(folderpath, 'mass_option' + str(opt) + '.dat')
    output = np.array(mass)
    df = pd.DataFrame(output, columns=['mass'])
    df.to_csv(name, sep='\t', index=False)