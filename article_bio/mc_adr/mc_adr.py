# %%

from cosmos.utils.generate_surface_meshes import generate_sphere
from ngsolve import *
from ngsolve.webgui import Draw

mesh, _ = generate_sphere(maxh = 0.07, R = 1)

from cosmos.pdes.pde_adr_bnd import BndADR
from cosmos.pdes.pde_mc_bgn_stab import MCBGNStab
from cosmos.pdes.pde_mc_bgn import MCBGN
# from cosmos.pdes.pde_mc_bgn_mod import MCBGNMod
from cosmos.solvers.time_schemes import BDF1, BDF2
from cosmos.pdes.pde_tools import SaveSolution
import numpy as np

import os
folderpath = './adr_mc'
os.makedirs(folderpath, exist_ok = True)
# Parameters
gamma = 100
a = 0.1
b = 0.9
delta = 0.4
dt = 1e-3

'''
First part where only the ADR equations are evolved
'''

T = 5
n = 100
sample_rate = np.maximum(int(T/dt/n), 1)
dt0 = Parameter(dt)
t0 = Parameter(0)

###################  SURFACE REACTIONS  ##################################
u0 = BndADR(d = 1, time_scheme=BDF1())
sol_save = SaveSolution(folderpath = folderpath,
                        filename = 'mc_adr_u0',
                        sample_rate = sample_rate)
u0.SaveSol(sol_save)
w0 = BndADR(d = 10, time_scheme=BDF1())
sol_save = SaveSolution(folderpath = folderpath,
                        filename = 'mc_adr_w0',
                        sample_rate = sample_rate)
w0.SaveSol(sol_save)

###################  Solver  ##################################
from cosmos.solvers.solvers import Dynamic
solver = Dynamic(mesh = mesh, t = t0, T = T, dt = dt0)

###################  ALE  ##################################

###################  Couplings  ##################################
from cosmos.pdes.coupling_weak import WeakCoupling
cpl = WeakCoupling(tol = 1e-5, type = 'implicit')
cpl.AddPDEs(u0, w0)
solver.AddPDE(cpl)

def u_rhs():
    term1 = gamma*(a-u0.solute+u0.solute**2*w0.solute)
    return term1
u0.rhs.value = u_rhs
def w_rhs():
    term1 = gamma*(b-u0.solute**2*w0.solute)
    return term1
w0.rhs.value = w_rhs

np.random.seed(42)
scene = Draw(u0.solute, mesh, deformation = solver.ale.deformation)
for i, sol in enumerate(solver()):
    if i == 0:
        n = len(u0.solute.vec.data)
        epsilon1 = np.random.uniform(0, 0.01, n)
        epsilon2 = np.random.uniform(0, 0.01, n)
        u0.solute.vec.data = a+b+epsilon1
        w0.solute.vec.data = b/(a+b)**2 + epsilon2

    scene.Redraw()

'''
Second part, which includes the mean curvature flow
'''

T = 2
n = 200
sample_rate = np.maximum(int(T/dt/n), 1)
dt = Parameter(dt)
t = Parameter(0)

###################  SURFACE REACTIONS  ##################################
u = BndADR(d = 1, time_scheme=BDF1())
sol_save = SaveSolution(folderpath = folderpath,
                        filename = 'mc_adr_u',
                        sample_rate = sample_rate)
u.SaveSol(sol_save)
w = BndADR(d = 10, time_scheme=BDF1())
sol_save = SaveSolution(folderpath = folderpath,
                        filename = 'mc_adr_w',
                        sample_rate = sample_rate)
w.SaveSol(sol_save)


###################  MEAN CURVATURE FLOW  ##################################
mc = MCBGNStab(time_scheme=BDF1(), postprocess=True, kappa=0.01)
sol_save = SaveSolution(folderpath = folderpath,
                        filename = 'mc_adr_mc',
                        sample_rate = sample_rate)
mc.SaveSol(sol_save)

###################  Solver  ##################################
from cosmos.solvers.solvers import Dynamic
solver = Dynamic(mesh = mesh, t = t, T = T, dt = dt)

###################  ALE  ##################################

###################  Couplings  ##################################
from cosmos.pdes.coupling_weak import WeakCoupling
cpl = WeakCoupling(tol = 1e-5, type = 'implicit')
cpl.AddPDEs(mc, u, w)
solver.AddPDE(cpl)

solver.ale.deformation_field = lambda: mc.displacement
solver.ale.velocity_field = lambda : mc.displacement/solver.dt
solver.ale.mat_velocity_field = lambda : mc.displacement/solver.dt

def u_rhs():
    term1 = gamma*(a-u.solute+u.solute**2*w.solute)
    return term1
u.rhs.value = u_rhs
def w_rhs():
    term1 = gamma*(b-u.solute**2*w.solute)
    return term1
w.rhs.value = w_rhs
def mc_rhs():
    n = specialcf.normal(3)
    term1 = delta*u.solute*n
    return term1
mc.rhs.value = mc_rhs

scene = Draw(u.solute, mesh, deformation = solver.ale.deformation)
for i, sol in enumerate(solver()):
    if i == 0:
        u.solute.vec.data = u0.solute.vec.data
        w.solute.vec.data = w0.solute.vec.data

    scene.Redraw()
# %%

# %%
