# %%

import numpy as np
from cosmos.utils.generate_surface_meshes import generate_half_sphere
from illposed_adr import solve_illposed_adr
import pandas as pd
import os
import matplotlib.pyplot as plt
from ngsolve import *

dt = 0.01
mesh, _ = generate_half_sphere(maxh = 0.1)
dts = np.arange(0.0, 1+dt, dt)

options = [0, 1, 2, 3]
folderpath = './illposed'
mass = []

for opt in options:
    name = 'illposed'
    mass_i = solve_illposed_adr(mesh=mesh, dt=dt, folderpath=folderpath, filename=name, option=opt)
    mass.append(mass_i)

mass = np.array(mass).transpose()
output = np.column_stack((dts, mass))

name = os.path.join(folderpath, 'mass.dat')
df = pd.DataFrame(output, columns=['dt', 'Solver1', 'Solver2', 'Solver3', 'Solver4'])
df.to_csv(name, sep='\t', index=False)

# mesh.UnsetDeformation()
# u_ex = exp(-3*(x**2 + y**2))
# n_ex = CF((x, y, z))/Norm(CF((x, y, z)))
# P_ex = Id(3) - OuterProduct(n_ex, n_ex)
# b = P_ex*CF((z,0,-x))

# vtkout = VTKOutput(mesh, coefs=[u_ex, b], names = ['u0', 'v'], filename = 'u0')
# vtkout.Do(vb=BND)