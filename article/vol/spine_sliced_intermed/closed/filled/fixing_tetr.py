# %%

from ngsolve import *
from ngsolve.webgui import Draw

from cosmos.utils.mesh_fixing import CosmosAliasMesh

name = "./spine_intermed_cut_fixed.vol"
mesh1 = Mesh("./spine_intermed_cut.vol")


mesh1 = CosmosAliasMesh("./spine_intermed_cut.vol")
mesh1.build_surface_from_volume()
mesh1.reorient_surface_triangles_consistently(flip=True)
mesh1.export_mesh(name)

mesh2 = Mesh(name)
print(mesh2.ne)
print(mesh2.nv)
print(mesh2.nface)
print(mesh2.nfacet)
print(mesh2.GetMaterials())
print(mesh2.GetBoundaries())
print(mesh2.GetBBoundaries())
print(mesh2.GetBBBoundaries())
Draw(mesh2)


mesh2 = Mesh("../../open/spine_intermed_sliced_PM_fixed.vol")
print(mesh2.ne)
print(mesh2.nv)
print(mesh2.nface)
print(mesh2.nfacet)
print(mesh2.GetMaterials())
print(mesh2.GetBoundaries())
print(mesh2.GetBBoundaries())
print(mesh2.GetBBBoundaries())
Draw(mesh2)

mesh1 = CosmosAliasMesh(name)

aux_mesh = CosmosAliasMesh("../../open/spine_intermed_sliced_PM_fixed.vol")
mesh1.mark_cd_elements(aux_mesh)
mesh1.export_mesh(name)

post_mesh = Mesh(name)
Draw(post_mesh)
print(post_mesh.ne)
print(post_mesh.nv)
print(post_mesh.nface)
print(post_mesh.nfacet)
print(post_mesh.GetMaterials())
print(post_mesh.GetBoundaries())
print(post_mesh.GetBBoundaries())

vtk = VTKOutput(post_mesh, filename="spine_intermed_cut_fixed")
vtk.Do()

# %%

# %%
