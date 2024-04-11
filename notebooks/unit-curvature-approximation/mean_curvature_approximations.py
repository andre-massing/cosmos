# %% Import relevant modules
from ngsolve import *
from netgen.occ import *
from ngsolve.webgui import Draw
ngsglobals.msg_level = 2

from cosmos.utils.generate_surface_meshes import *
from cosmos.solvers.curvature_calculators import *

# %% Define geometry and mesh it
# shape = "sphere"
shape = "sphere"
order_g = 1
maxh = 0.1

if shape == "sphere" :
    R = 1
    sphere = Sphere((0,0,0),R).faces[0]
    geo = OCCGeometry(sphere)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    mesh.Curve(order_g)
else :
    # Mesh geometry with curved elements of order 2
    # Torus mesh
    mesh = generate_torus_mesh(maxh=maxh)
    mesh.Curve(order_g)
    Tend = 0.11

Draw(mesh)

# %% Compute mean curvature
kappah = compute_mean_curvature_vector(mesh)
Draw(kappah, mesh)

# %%
# With this choice of the stabilization parameter, the norm of discrete mean curvature is almost exact 2 for the unit sphere
gamma_E = 7.0e-3
kappah_stab = compute_stabilized_mean_curvature_vector(mesh, gamma_E)
Draw(kappah_stab, mesh)

# %%
