# %%

from cosmos.utils.generate_surface_meshes import generate_circle
from ngsolve import *
from ngsolve.webgui import Draw
from netgen.webgui import Draw as DrawGeo
import netgen.occ as occ

R = 1
maxh = 0.08
angle = pi/6  # 30 degrees

pnt1 = occ.Pnt(0, 0, R)
pnt2 = occ.Pnt(R*sin(angle/2), 0, R*cos(angle/2))
pnt3 = occ.Pnt(R*sin(angle) , 0, R*cos(angle))
pnt4 = occ.Pnt(R*sin(angle*2) , 0, R*cos(angle*2))
pnt5 = occ.Pnt(0 , 0, -R)

arc1 = occ.ArcOfCircle(pnt1, pnt2, pnt3)
arc2 = occ.ArcOfCircle(pnt3, pnt4, pnt5)
seg1 = occ.Segment(pnt1, pnt5)

w = occ.Wire([arc1, arc2, seg1])
w.edges[0].name = 'bnd1'
w.edges[1].name = 'bnd2'
# w.edges[0].maxh = 0.1
# w.edges[1].maxh = 0.1
f = occ.Face(w)
body = f.Revolve(occ.Axis((0,0,0),occ.Z), 360).Rotate(occ.Axis((0,0,0),occ.Y), 90)

geo = occ.OCCGeometry(body)
ngmesh = geo.GenerateMesh(maxh=maxh, optsteps2d=3)
mesh = Mesh(ngmesh)

Draw(mesh)

print(mesh.ne)
print(mesh.nv)

# %%


from cosmos.pdes.pde_adr_vol import VolADR
from cosmos.pdes.pde_mc_bgn_stab import MCBGNStab
from cosmos.pdes.pde_mc_bgn import MCBGN
from cosmos.pdes.pde_willmore_dziuk import WillmoreDziuk
# from cosmos.pdes.pde_mc_bgn_mod import MCBGNMod
from cosmos.solvers.time_schemes import BDF1, BDF2
from cosmos.pdes.pde_tools import SaveSolution
import numpy as np

import os
folderpath = './herant_3d'
os.makedirs(folderpath, exist_ok = True)

T = 3
dt = 2e-3
n = 200
sample_rate = np.maximum(int(T/dt/n), 1)
dt = Parameter(dt)
t = Parameter(0)

ns = specialcf.normal(3)
neu_d = {'bnd1': -1*ns}
###################  SURFACE REACTIONS  ##################################
u = VolADR(d = 1, c= 1, neu_d= neu_d,  time_scheme=BDF1())
sol_save = SaveSolution(folderpath = folderpath,
                        filename = 'herant_3d_u',
                        sample_rate = sample_rate)
u.SaveSol(sol_save)

###################  MEAN CURVATURE FLOW  ##################################
# mc = MCBGNStab(time_scheme=BDF1(), postprocess=True, alpha=0.01)
mc = WillmoreDziuk(time_scheme=BDF1(), postprocess=True, kappa=0.01)
sol_save = SaveSolution(folderpath = folderpath,
                        filename = 'herant_3d_mc',
                        sample_rate = sample_rate)
mc.SaveSol(sol_save)

###################  Solver  ##################################
from cosmos.solvers.solvers import Dynamic
solver = Dynamic(mesh = mesh, t = t, T = T, dt = dt)

###################  ALE  ##################################

###################  Couplings  ##################################
from cosmos.pdes.coupling_weak import WeakCoupling
cpl = WeakCoupling(tol = 1e-5, type = 'explicit')
cpl.AddPDEs(u, mc)
solver.AddPDE(cpl)

F0 = 1
mc.rhs.value = lambda: F0*u.solute*ns
solver.ale.deformation_field = lambda: mc.displacement
solver.ale.velocity_field = lambda : mc.displacement/dt
solver.ale.mat_velocity_field = lambda : mc.displacement/dt


scene = Draw(u.solute, mesh, deformation = solver.ale.deformation)
for i, sol in enumerate(solver()):
    scene.Redraw()
# %%

# %%
