# %% Import relevant modules
from ngsolve import *
from netgen.occ import *
from ngsolve.webgui import Draw
ngsglobals.msg_level = 1
import numpy as np

# %% Define geometry and mesh it
R = 1
sphere = Sphere((0,0,0),R).faces[0]

# Mesh geometry with curved elements of order 2
order_g = 2
geo = OCCGeometry(sphere)

maxh = 0.05
mesh = Mesh(geo.GenerateMesh(maxh=maxh))
mesh.Curve(order_g)
Draw(mesh)

#%%[markdown] Define mixed function space for parametrization and curvature
# $\int_a^b f(x)dx$

# %% Define mixed function space for parametrization and curvature
order_p = order_g
V = VectorH1(mesh, order=order_p)
Q = H1(mesh, order=order_p)
W = V*Q

# Displacement to deform initial mesh into final mesh
# TODO: Check whether we can simply take dXh = GridFunction(V)
#       instead of full mixed space W
gfW = GridFunction(W)
dXh = gfW.components[0]
dXh.vec[:] = 0

#%%
# Combined parameter mapping and discrete mean curvature as grid function
Xkappah = GridFunction(W)
Xh, kappah = Xkappah.components
# Setting Inital deformation
Xh.Set( CF( (x,y,z) ), definedon=mesh.Boundaries(".*"))
kappah.Set(CF(2/R),    definedon=mesh.Boundaries(".*"))

# # Store identity mapping as grid function,
# to be substracted from parameter mapping to compute 
# displacement of initial mesh
Idh = GridFunction(V)
Idh.Set( CF( (x,y,z) ), definedon=mesh.Boundaries(".*"))

diff = Idh.vec.CreateVector()

Xh.vec.data -= Idh.vec
np.abs(Xh.vec.FV().NumPy()).max()
# print(Xh.vec)
# %%
