# %%
# Importing general libraries

from cosmos.solvers.solvers import Dynamic
from cosmos.solvers.time_schemes import BDF1, BDF2
from cosmos.pdes.pde_willmore_dziuk import WillmoreDziuk
from ngsolve import *
from ngsolve.webgui import Draw
from cosmos.utils.generate_surface_meshes import generate_box
from cosmos.pdes.pde_tools import SaveSolution
import numpy as np
import pandas as pd

folderpath = './tang_nostab'

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

def willmore_dziuk_tang_nostab(mesh, dt, T, filename):

    t = Parameter(0.0)
    dt = Parameter(dt)

    # With boundary conditions
    sp_curv = -3
    willmore = WillmoreDziuk(name = [filename + 'dX', filename + 'mc'],
                           time_scheme=BDF1(), postprocess=True,
                           mc_autoupdate=True, sp_curv=sp_curv)
    sol_save = SaveSolution(folderpath = folderpath,
                            filename = filename,
                            sample_rate = 50)
    willmore.SaveSol(sol_save)

    solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)
    solver.AddPDE(willmore)
    solver.ale.deformation_field = willmore.gfu.components[0]

    tang_nostab = []
    time = []
    ns = specialcf.normal(3)
    try:
        for sol in solver():
            tang_nostab_i = 0.5*Integrate(Norm(willmore.mean_curvature - ns*sp_curv)**2, mesh, VOL_or_BND=BND)
            tang_nostab.append(tang_nostab_i)
            time.append(t.Get())
    except:
        print('Simulation terminated with error')
    
    tang_nostab = np.array(tang_nostab)
    time = np.array(time)
    return time, tang_nostab

time, tang_nostab = willmore_dziuk_tang_nostab(mesh=mesh, dt=dt, T=T, filename=name)

n = 200
if len(time)>n:
    indices = np.linspace(0, len(time) - 1, n, dtype=int)
    tang_nostab = tang_nostab[indices]
    time = time[indices]

df = pd.DataFrame(np.column_stack([time, tang_nostab]), columns = ['Time',  name])
df.to_csv(os.path.join(folderpath,name + '.dat'), sep='\t', index=False)

# %%
# Case of box 3x1x1 with spontaneous curvature -3 

name = 'box311'
mesh, _ = generate_box(a=1, b=3, c=1, maxh=0.15, vol_or_bnd='BND')
print(mesh.nfacet)
print(mesh.nv)
Draw(mesh)
dt = 1e-4
T = 0.5


os.makedirs(folderpath, exist_ok=True)

def willmore_dziuk_tang_nostab(mesh, dt, T, filename):

    t = Parameter(0.0)
    dt = Parameter(dt)

    # With boundary conditions
    sp_curv = -2
    willmore = WillmoreDziuk(name = [filename + 'dX', filename + 'mc'],
                           time_scheme=BDF1(), postprocess=True,
                           mc_autoupdate=True, sp_curv=sp_curv)
    sol_save = SaveSolution(folderpath = folderpath,
                            filename = filename,
                            sample_rate = 50)
    willmore.SaveSol(sol_save)

    solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)
    solver.AddPDE(willmore)
    solver.ale.deformation_field = willmore.gfu.components[0]

    tang_nostab = []
    time = []
    ns = specialcf.normal(3)
    try:
        for sol in solver():
            tang_nostab_i = 0.5*Integrate(Norm(willmore.mean_curvature - ns*sp_curv)**2, mesh, VOL_or_BND=BND)
            tang_nostab.append(tang_nostab_i)
            time.append(t.Get())
    except:
        print('Simulation terminated with error')
    
    tang_nostab = np.array(tang_nostab)
    time = np.array(time)
    return time, tang_nostab

time, tang_nostab = willmore_dziuk_tang_nostab(mesh=mesh, dt=dt, T=T, filename=name)

n = 200
if len(time)>n:
    indices = np.linspace(0, len(time) - 1, n, dtype=int)
    tang_nostab = tang_nostab[indices]
    time = time[indices]

df = pd.DataFrame(np.column_stack([time, tang_nostab]), columns = ['Time',  name])
df.to_csv(os.path.join(folderpath,name + '.dat'), sep='\t', index=False)
# %%
