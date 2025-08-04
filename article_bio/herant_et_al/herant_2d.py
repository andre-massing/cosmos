# %%

from cosmos.utils.generate_surface_meshes import generate_circle
from ngsolve import *
from ngsolve.webgui import Draw
from netgen.webgui import Draw as DrawGeo
import netgen.occ as occ

R = 1
maxh = 0.1
angle = pi/6  # 30 degrees

pnt1 = occ.Pnt(R, 0, 0)
pnt2 = occ.Pnt(R*cos(angle), R*sin(angle), 0)
pnt3 = occ.Pnt(R*cos(angle), -R*sin(angle), 0)
pnt4 = occ.Pnt(-R, 0, 0)

arc1 = occ.ArcOfCircle(pnt2, pnt1, pnt3)
arc2 = occ.ArcOfCircle(pnt3, pnt4, pnt2)

w = occ.Wire([arc1, arc2])
f = occ.Face(w)

geo = occ.OCCGeometry(f, dim = 2)
ngmesh = geo.GenerateMesh(maxh=maxh)
ngmesh.SetBCName(0, 'bnd1')
ngmesh.SetBCName(1, 'bnd2')
mesh = Mesh(ngmesh)

Draw(mesh)
print(mesh.ne)
print(mesh.nv)


from cosmos.pdes.pde_adr_vol import VolADR
from cosmos.pdes.pde_mc_bgn_stab import MCBGNStab
from cosmos.pdes.pde_mc_bgn import MCBGN
from cosmos.pdes.pde_willmore_dziuk import WillmoreDziuk
from cosmos.solvers.time_schemes import BDF1, BDF2
from cosmos.pdes.pde_tools import SaveSolution
import numpy as np

import os
folderpath = './herant_2d'
os.makedirs(folderpath, exist_ok = True)

T = 3
dt = 2e-3
n = 200
sample_rate = np.maximum(int(T/dt/n), 1)
dt = Parameter(dt)
t = Parameter(0)

ns = specialcf.normal(2)
neu_d = {'bnd1': -1*ns}
###################  SURFACE REACTIONS  ##################################
u = VolADR(d = 1, c= 1, neu_d= neu_d,  time_scheme=BDF1())
sol_save = SaveSolution(folderpath = folderpath,
                        filename = 'herant_2d_u',
                        sample_rate = sample_rate)
u.SaveSol(sol_save)

###################  MEAN CURVATURE FLOW  ##################################
# mc = MCBGN(time_scheme=BDF1(), postprocess=True, alpha=0.01)
mc = WillmoreDziuk(time_scheme=BDF1(), postprocess=True, kappa=1)
sol_save = SaveSolution(folderpath = folderpath,
                        filename = 'herant_2d_mc',
                        sample_rate = sample_rate)
mc.SaveSol(sol_save)

###################  Solver  ##################################
from cosmos.solvers.solvers import Dynamic
solver = Dynamic(mesh = mesh, t = t, T = T, dt = dt)

###################  ALE  ##################################

###################  Couplings  ##################################
from cosmos.pdes.coupling_weak import WeakCoupling
cpl = WeakCoupling(tol = 1e-5, type = 'implicit')
cpl.AddPDEs(u, mc)
solver.AddPDE(cpl)

F0 = -1
mc.rhs.value = lambda: F0*u.solute*ns
solver.ale.deformation_field = lambda: mc.displacement
solver.ale.velocity_field = lambda : mc.displacement/dt
solver.ale.mat_velocity_field = lambda : mc.displacement/dt


scene1 = Draw(u.solute, mesh, deformation = solver.ale.deformation)
scene2 = Draw(solver.ale.deformation, mesh, deformation = solver.ale.deformation)
for i, sol in enumerate(solver()):
    scene1.Redraw()
    scene2.Redraw()
# %%

# %%
