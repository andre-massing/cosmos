# %%
# Importing general libraries

from cosmos.solvers.solvers import Dynamic
from cosmos.solvers.time_schemes import BDF1, BDF2
from cosmos.pdes.pde_willmore_dziuk_stab import WillmoreDziukStab
from ngsolve import *
from ngsolve.webgui import Draw
from cosmos.utils.generate_surface_meshes import generate_torus
from cosmos.pdes.pde_tools import SaveSolution
import numpy as np
import pandas as pd

folderpath = './tang_stab'

# %%
# Case NgSolve generated torus

name = 'ngsolve_torus'
mesh, _ = generate_torus(R=2, r=1, maxh = 0.22, vol_or_bnd='BND')
print(mesh.nfacet)
print(mesh.nv)
Draw(mesh)
dt = 2e-4
T = 2

os.makedirs(folderpath, exist_ok=True)

def willmore_dziuk_tang_stab(mesh, dt, T, filename):

    t = Parameter(0.0)
    dt = Parameter(dt)

    # With boundary conditions
    willmore = WillmoreDziukStab(name = [filename + 'dX', filename + 'mc'],
                           time_scheme=BDF1(), postprocess=True,
                           mc_autoupdate=True)
    sol_save = SaveSolution(folderpath = folderpath,
                            filename = filename,
                            sample_rate = 50)
    willmore.SaveSol(sol_save)

    solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)
    solver.AddPDE(willmore)
    solver.ale.deformation_field = willmore.gfu.components[0]

    tang_stab = []
    time = []
    ns = specialcf.normal(3)
    try:
        for sol in solver():
            tang_stab_i = 0.5*Integrate(Norm(willmore.mean_curvature)**2, mesh, VOL_or_BND=BND)
            tang_stab.append(tang_stab_i)
            time.append(t.Get())
    except:
        print('Simulation terminated with error')
    
    tang_stab = np.array(tang_stab)
    time = np.array(time)
    return time, tang_stab

time, tang_stab = willmore_dziuk_tang_stab(mesh=mesh, dt=dt, T=T, filename=name)

n = 200
if len(time)>n:
    indices = np.linspace(0, len(time) - 1, n, dtype=int)
    tang_stab = tang_stab[indices]
    time = time[indices]

df = pd.DataFrame(np.column_stack([time, tang_stab]), columns = ['Time',  name])
df.to_csv(os.path.join(folderpath,name + '.dat'), sep='\t', index=False)

# %%
# Case MeshLab Solve generated torus 
name = 'meshlab_torus'

mesh = Mesh('../../meshes/torus.vol')
print(mesh.nfacet)
print(mesh.nv)
Draw(mesh)
dt = 2e-4
T = 2

os.makedirs(folderpath, exist_ok=True)

def willmore_dziuk_tang_stab(mesh, dt, T, filename):

    t = Parameter(0.0)
    dt = Parameter(dt)

    # With boundary conditions
    willmore = WillmoreDziukStab(name = [filename + 'dX', filename + 'mc'],
                           time_scheme=BDF1(), postprocess=True,
                           mc_autoupdate=True)
    sol_save = SaveSolution(folderpath = folderpath,
                            filename = filename,
                            sample_rate = 50)
    willmore.SaveSol(sol_save)

    solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)
    solver.AddPDE(willmore)
    solver.ale.deformation_field = willmore.gfu.components[0]

    tang_stab = []
    time = []
    ns = specialcf.normal(3)
    try:
        for sol in solver():
            tang_stab_i = 0.5*Integrate(Norm(willmore.mean_curvature)**2, mesh, VOL_or_BND=BND)
            tang_stab.append(tang_stab_i)
            time.append(t.Get())
    except:
        print('Simulation terminated with error')
    
    tang_stab = np.array(tang_stab)
    time = np.array(time)
    return time, tang_stab

time, tang_stab = willmore_dziuk_tang_stab(mesh=mesh, dt=dt, T=T, filename=name)

n = 200
if len(time)>n:
    indices = np.linspace(0, len(time) - 1, n, dtype=int)
    tang_stab = tang_stab[indices]
    time = time[indices]

df = pd.DataFrame(np.column_stack([time, tang_stab]), columns = ['Time',  name])
df.to_csv(os.path.join(folderpath,name + '.dat'), sep='\t', index=False)
# %%
