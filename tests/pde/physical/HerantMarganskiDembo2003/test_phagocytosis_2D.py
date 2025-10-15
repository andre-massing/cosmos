# %%

from cosmos.utils.generate_meshes import generate_boundary_circle
from ngsolve import *
from ngsolve.webgui import Draw
from netgen.webgui import Draw as DrawGeo
import netgen.occ as occ
from cosmos import *

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


import numpy as np


T = 3
dt = 2e-3
n = 200
sample_rate = np.maximum(int(T/dt/n), 1)
dt = Parameter(dt)
t = Parameter(0)

###################  Solver  ##################################
solvermesh = SolverMesh(mesh)
solvertime = SolverTime(dt = dt, initial_t=0, final_t=T)
solver = Solver(solvermesh, solvertime, iter=True,
                printing=True)

###################  SURFACE REACTIONS  ##################################
ns = specialcf.normal(2)
u = ADRVolumeBDF1Model(solver, 1, input_params={'Neu_bnd': 'bnd1'})
u.set_input_fields({
    'd': 1,
    'c': 1,
    'gradu_bnd': ns
})

################### GRADIENT FLOW  ##################################
# mc = MeanCurvatureBoundaryBDF1Model(solver, 2, input_params={'kappa': 0.01})
# F0 = -1
# mc.set_input_fields({
#     'rhs': lambda: F0*u.sol*ns
# })
willmore = WillmoreBoundaryInexBDF1Model(solver, 2)
F0 = -1
willmore.set_input_fields({
    'rhs': lambda: F0*u.sol*ns,
    'elasticity_modulus': 1e-3

})

###################  ALE  ##################################
ale = ALEModel(solver, 3)
ale.set_bnd_displacement(willmore.displacement, 'bnd1|bnd2', redistribute=True, redistribute_type='DuanLi')


scene1 = Draw(u.sol, mesh, deformation = ale.displacement)
scene2 = Draw(ale.displacement, mesh, deformation = ale.displacement)
for i, sol in enumerate(solver()):
    scene1.Redraw()
    scene2.Redraw()

# %%
