# %%

from cosmos.utils.generate_surface_meshes import generate_sphere, generate_circle
from ngsolve import *
from ngsolve.webgui import Draw

R = 1
mesh, _ = generate_sphere(maxh = 0.15, R = R)
mesh, _ = generate_circle(maxh = 0.1, R = R)
Area0 = Integrate(1, mesh, VOL_or_BND=BND)

from cosmos.pdes.pde_adr_bnd import BndADR

from cosmos.solvers.time_schemes import BDF1, BDF2
from cosmos.pdes.pde_tools import SaveSolution
import numpy as np

import os
folderpath = './lomakin_2d'
os.makedirs(folderpath, exist_ok = True)

T = 100
dt = 1e-2
n = 200
sample_rate = np.maximum(int(T/dt/n), 1)
dt = Parameter(dt)
t = Parameter(0)
a_12 = 1
a_21 = 1
kappa1 = 0.4
kappa2 = 0.4
m = 4
s = 8

###################  SURFACE REACTIONS  ##################################
u = BndADR(u0 = 0.5, d = 0.1, c = -1, time_scheme=BDF1())
sol_save = SaveSolution(folderpath = folderpath,
                        filename = 'lomakin_u',
                        sample_rate = sample_rate)
u.SaveSol(sol_save)
angle = sqrt(3)/2
u0 = IfPos(x/R - angle, 0.1, 0.5)
w = BndADR(u0 = u0, d = 0.1, c = -1, time_scheme=BDF1())
sol_save = SaveSolution(folderpath = folderpath,
                        filename = 'lomakin_w',
                        sample_rate = sample_rate)
w.SaveSol(sol_save)

###################  Solver  ##################################
from cosmos.solvers.solvers import Dynamic
solver = Dynamic(mesh = mesh, t = t, T = T, dt = dt)

###################  ALE  ##################################

###################  Couplings  ##################################
from cosmos.pdes.coupling_weak import WeakCoupling
cpl = WeakCoupling(tol = 1e-10, type = 'implicit')
cpl.AddPDEs(u, w)
solver.AddPDE(cpl)

def V():
    solver.mesh.SetDeformation(solver.ale.deformation)
    Area = Integrate(1, mesh, VOL_or_BND=BND)
    stretch = Area/Area0
    term1 = u.solute**m/(u.solute**m + stretch**m)
    term2 = w.solute**m/(w.solute**m + 1)
    solver.mesh.UnsetDeformation()

    return term1 - term2

ns = specialcf.normal(mesh.dim)
solver.ale.deformation_field = lambda: V()*dt*ns
solver.ale.velocity_field = lambda : V()*ns
solver.ale.mat_velocity_field = lambda : V()*ns

def u_rhs():
    term1 = kappa1*V()*u.solute - u.solute**2 - a_12*u.solute*w.solute
    return term1
u.rhs.value = u_rhs
def w_rhs():
    term1 = -kappa2*V()*w.solute - w.solute**2 - a_21*u.solute*w.solute
    return term1
w.rhs.value = w_rhs

# np.random.seed(42)
scene = Draw(w.gfu_save[0], mesh, deformation = solver.ale.deformation)
for i, sol in enumerate(solver()):
    # if i == 0:
    #     n = len(u.solute.vec.data)
    #     epsilon1 = np.random.uniform(0, 0.01, n)
    #     epsilon2 = np.random.uniform(0, 0.01, n)
    #     u.solute.vec.data = a+b+epsilon1
    #     w.solute.vec.data = b/(a+b)**2 + epsilon2

    scene.Redraw()

# %%
