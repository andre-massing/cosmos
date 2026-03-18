# %%

from ngsolve import *
from ngsolve.webgui import Draw
from cosmos.utils.mesh_fixing import CosmosAliasMesh
from cosmos.utils.mesh_fixing import fill_mesh

mesh1 = Mesh('../spine_coarseNEW_sliced_PM_closed_fixed.vol')
mesh1 = fill_mesh(mesh1, maxh = 0.05)
name = 'spine_coarseNEW_sliced_PM_closed_filled_fixed.vol'
mesh1.ngmesh.Save(name)
mesh1 = Mesh(name)
print(mesh1.ne)
print(mesh1.GetMaterials())
print(mesh1.GetBoundaries())
print(mesh1.GetBBoundaries())
print(mesh1.GetBBBoundaries())
Draw(mesh1)

mesh2 = Mesh('../../open/spine_coarseNEW_sliced_PM_fixed.vol')
print(mesh2.ne)
print(mesh2.GetMaterials())
print(mesh2.GetBoundaries())
print(mesh2.GetBBoundaries())
print(mesh2.GetBBBoundaries())
Draw(mesh2)

fixed_mesh = CosmosAliasMesh(name)
aux_mesh = CosmosAliasMesh('../../open/spine_coarseNEW_sliced_PM_fixed.vol')
fixed_mesh.mark_cd_elements(aux_mesh)
fixed_mesh.export_mesh(name)

post_mesh = Mesh(name)
print(post_mesh.ne)
print(post_mesh.GetMaterials())
print(post_mesh.GetBoundaries())
print(post_mesh.GetBBoundaries())
print(post_mesh.GetBBBoundaries())

gfu = GridFunction(H1(post_mesh))
cf = post_mesh.BoundaryCF({
    'default': CF(1),
    'boundary2': CF(2)
})
gfu.Set(cf, definedon = post_mesh.Boundaries('.*'))
Draw(gfu, post_mesh)

vtk = VTKOutput(post_mesh, filename = 'meshNEW')
vtk.Do()