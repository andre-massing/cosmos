# %%

from ngsolve import *
from ngsolve.webgui import Draw

from cosmos.utils.mesh_fixing import CosmosAliasMesh

mesh = Mesh("spine_sliced_PM_closed.vol")
print(mesh.GetMaterials())
print(mesh.GetBoundaries())
print(mesh.GetBBoundaries())
print(mesh.GetBBBoundaries())
Draw(mesh)

fixed_mesh = CosmosAliasMesh("spine_sliced_PM_closed.vol")
fixed_mesh.reorient_surface_triangles_consistently()
name = "spine_coarse_sliced_PM_closed_fixed.vol"
fixed_mesh.export_mesh(name)

post_mesh = Mesh(name)
print(post_mesh.GetMaterials())
print(post_mesh.GetBoundaries())
print(post_mesh.GetBBoundaries())
print(post_mesh.GetBBBoundaries())
Draw(post_mesh)
