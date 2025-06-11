# %%
# Importing general libraries

from cosmos.solvers.solvers import Dynamic
from cosmos.solvers.time_schemes import BDF1, BDF2
from cosmos.pdes.pde_willmore_dziuk_stab import WillmoreDziukStab
from ngsolve import *
from ngsolve.webgui import Draw
from cosmos.utils.generate_surface_meshes import generate_box
from cosmos.pdes.pde_tools import SaveSolution
import numpy as np
import pandas as pd

folderpath = './notang_stab'

# %%
# Case of box 6x1x1 with spontaneous curvature -3 

name = 'box611'
mesh, _ = generate_box(a=1, b=5, c=1, maxh=0.15, vol_or_bnd='BND')
print(mesh.nfacet)
print(mesh.nv)
Draw(mesh)
dt = 1e-4
T = 0.5

os.makedirs(folderpath, exist_ok=True)

def willmore_dziuk_notang_stab(mesh, dt, T, filename):

    t = Parameter(0.0)
    dt = Parameter(dt)

    # With boundary conditions
    sp_curv = -3
    willmore = WillmoreDziukStab(name = [filename + 'dX', filename + 'mc'],
                           time_scheme=BDF1(), sp_curv=sp_curv)
    sol_save = SaveSolution(folderpath = folderpath,
                            filename = filename,
                            sample_rate = 50)
    willmore.SaveSol(sol_save)

    solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)
    solver.AddPDE(willmore)
    solver.ale.deformation_field = willmore.gfu.components[0]

    notang_stab = []
    time = []
    ns = specialcf.normal(3)
    try:
        for sol in solver():
            notang_stab_i = 0.5*Integrate(Norm(willmore.mean_curvature - ns*sp_curv)**2, mesh, VOL_or_BND=BND)
            notang_stab.append(notang_stab_i)
            time.append(t.Get())
    except:
        print('Simulation terminated with error')
    
    notang_stab = np.array(notang_stab)
    time = np.array(time)
    return time, notang_stab

time, notang_stab = willmore_dziuk_notang_stab(mesh=mesh, dt=dt, T=T, filename=name)

n = 200
if len(time)>n:
    indices = np.linspace(0, len(time) - 1, n, dtype=int)
    notang_stab = notang_stab[indices]
    time = time[indices]

df = pd.DataFrame(np.column_stack([time, notang_stab]), columns = ['Time',  name])
df.to_csv(os.path.join(folderpath,name + '.dat'), sep='\t', index=False)

# %%
# Case of box 3x1x1 with spontaneous curvature -2 
name = 'box311'
mesh, _ = generate_box(a=1, b=3, c=1, maxh=0.15, vol_or_bnd='BND')
print(mesh.nfacet)
print(mesh.nv)
Draw(mesh)
dt = 1e-4
T = 0.5

def willmore_dziuk_notang_stab(mesh, dt, T, filename):

    t = Parameter(0.0)
    dt = Parameter(dt)

    # With boundary conditions
    sp_curv = -2
    willmore = WillmoreDziukStab(name = [filename + 'dX', filename + 'mc'],
                           time_scheme=BDF1(), sp_curv=sp_curv)
    sol_save = SaveSolution(folderpath = folderpath,
                            filename = filename,
                            sample_rate = 50)
    willmore.SaveSol(sol_save)

    solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)
    solver.AddPDE(willmore)
    solver.ale.deformation_field = willmore.gfu.components[0]

    notang_stab = []
    time = []
    ns = specialcf.normal(3)
    try:
        for sol in solver():
            notang_stab_i = 0.5*Integrate(Norm(willmore.mean_curvature - ns*sp_curv)**2, mesh, VOL_or_BND=BND)
            notang_stab.append(notang_stab_i)
            time.append(t.Get())
    except:
        print('Simulation terminated with error')
    
    notang_stab = np.array(notang_stab)
    time = np.array(time)
    return time, notang_stab

time, notang_stab = willmore_dziuk_notang_stab(mesh=mesh, dt=dt, T=T, filename=name)

n = 200
if len(time)>n:
    indices = np.linspace(0, len(time) - 1, n, dtype=int)
    notang_stab = notang_stab[indices]
    time = time[indices]

df = pd.DataFrame(np.column_stack([time, notang_stab]), columns = ['Time',  name])
df.to_csv(os.path.join(folderpath, name + '.dat'), sep='\t', index=False)
# %%
