# %%

from ngsolve import *
from ngsolve.webgui import Draw
from cosmos.utils.mesh_fixing import CosmosAliasMesh

mesh1 = Mesh('../spine_coarse_PM_closed_fixed.vol')
print(mesh1.GetMaterials())
print(mesh1.GetBoundaries())
print(mesh1.GetBBoundaries())
print(mesh1.GetBBBoundaries())
Draw(mesh1)

mesh2 = Mesh('../../open/spine_coarse_PM_fixed.vol')
print(mesh2.GetMaterials())
print(mesh2.GetBoundaries())
print(mesh2.GetBBoundaries())
print(mesh2.GetBBBoundaries())
Draw(mesh2)

fixed_mesh = CosmosAliasMesh('../spine_coarse_PM_closed_fixed.vol')
aux_mesh = CosmosAliasMesh('../../open/spine_coarse_PM_fixed.vol')
fixed_mesh.mark_cd_elements(aux_mesh)
name = 'spine_coarse_PM_closed_with_holes_marked_fixed.vol'
fixed_mesh.export_mesh(name)

post_mesh = Mesh(name)
print(post_mesh.GetMaterials())
print(post_mesh.GetBoundaries())
print(post_mesh.GetBBoundaries())
print(post_mesh.GetBBBoundaries())
cf = post_mesh.BoundaryCF({
    'default': CF(1),
    'boundary2': CF(2)
})
Draw(cf, post_mesh)

# %%

from ngsolve import *
from ngsolve.webgui import Draw
from cosmos.utils.mesh_fixing import CosmosAliasMesh

mesh1 = Mesh('../spine_fine_PM_closed_fixed.vol')
print(mesh1.GetMaterials())
print(mesh1.GetBoundaries())
print(mesh1.GetBBoundaries())
print(mesh1.GetBBBoundaries())
Draw(mesh1)

mesh2 = Mesh('../../open/spine_fine_PM_fixed.vol')
print(mesh2.GetMaterials())
print(mesh2.GetBoundaries())
print(mesh2.GetBBoundaries())
print(mesh2.GetBBBoundaries())
Draw(mesh2)

fixed_mesh = CosmosAliasMesh('../spine_fine_PM_closed_fixed.vol')
aux_mesh = CosmosAliasMesh('../../open/spine_fine_PM_fixed.vol')
fixed_mesh.mark_cd_elements(aux_mesh)
name = 'spine_fine_PM_closed_with_holes_marked_fixed.vol'
fixed_mesh.export_mesh(name)

post_mesh = Mesh(name)
print(post_mesh.GetMaterials())
print(post_mesh.GetBoundaries())
print(post_mesh.GetBBoundaries())
print(post_mesh.GetBBBoundaries())
cf = post_mesh.BoundaryCF({
    'default': CF(1),
    'boundary2': CF(2)
})
Draw(cf, post_mesh)