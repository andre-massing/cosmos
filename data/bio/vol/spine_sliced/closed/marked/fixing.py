# %%

from ngsolve import *
from ngsolve.webgui import Draw
from cosmos.utils.mesh_fixing import CosmosAliasMesh

mesh1 = Mesh('../spine_sliced_PM_closed_fixed.vol')
print(mesh1.GetMaterials())
print(mesh1.GetBoundaries())
print(mesh1.GetBBoundaries())
print(mesh1.GetBBBoundaries())
Draw(mesh1)

mesh2 = Mesh('../../open/with_holes/spine_sliced_PM_with_holes_fixed.vol')
print(mesh2.GetMaterials())
print(mesh2.GetBoundaries())
print(mesh2.GetBBoundaries())
print(mesh2.GetBBBoundaries())
Draw(mesh2)

fixed_mesh = CosmosAliasMesh('../spine_sliced_PM_closed_fixed.vol')
aux_mesh = CosmosAliasMesh('../../open/with_holes/spine_sliced_PM_with_holes_fixed.vol')
fixed_mesh.mark_cd_elements(aux_mesh)
name = 'spine_sliced_PM_closed_with_holes_marked_fixed.vol'
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

mesh1 = Mesh('../spine_sliced_PM_closed_fixed.vol')
print(mesh1.GetMaterials())
print(mesh1.GetBoundaries())
print(mesh1.GetBBoundaries())
print(mesh1.GetBBBoundaries())
Draw(mesh1)

mesh2 = Mesh('../../open/spine_sliced_PM_fixed.vol')
print(mesh2.GetMaterials())
print(mesh2.GetBoundaries())
print(mesh2.GetBBoundaries())
print(mesh2.GetBBBoundaries())
Draw(mesh2)

fixed_mesh = CosmosAliasMesh('../spine_sliced_PM_closed_fixed.vol')
aux_mesh = CosmosAliasMesh('../../open/spine_sliced_PM_fixed.vol')
fixed_mesh.mark_cd_elements(aux_mesh)
name = 'spine_sliced_PM_closed_marked_fixed.vol'
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