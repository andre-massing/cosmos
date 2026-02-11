# %%

from ngsolve import *
from ngsolve.webgui import Draw
import netgen.occ as occ
from cosmos import *
import numpy as np
import logging
import pytest
from cosmos.core.model import CosmosModel
from cosmos.pde.willmore.geometrical_flow_model import GeometricalFlowModel
from cosmos.pde.adr.volume.adr_volume_bdf1_model import ADRVolumeBDF1Model
from cosmos.pde.adr.volume.adr_volume_stab_bdf1_model import ADRVolumeStabBDF1Model

logging.getLogger().setLevel(logging.INFO)


R = 1
maxh = 0.2

angle1_deg = 20
angle1 = angle1_deg/180*pi

angle2_deg = 45
angle2 = angle2_deg/180*pi

pnt0 = occ.Pnt(R*cos(angle1), -R*sin(angle1), 0)
pnt1 = occ.Pnt(R, 0, 0)
pnt2 = occ.Pnt(R*cos(angle1), R*sin(angle1), 0)
pnt3 = occ.Pnt(0, R, 0)
pnt4 = occ.Pnt(-R*cos(angle2), R*sin(angle2), 0)
pnt5 = occ.Pnt(-R, 0, 0)
pnt6 = occ.Pnt(-R*cos(angle2), -R*sin(angle2), 0)
pnt7 = occ.Pnt(0, -R, 0)

arc1 = occ.ArcOfCircle(pnt0, pnt1, pnt2)
arc2 = occ.ArcOfCircle(pnt2, pnt3, pnt4)
arc3 = occ.ArcOfCircle(pnt4, pnt5, pnt6)
arc4 = occ.ArcOfCircle(pnt6, pnt7, pnt0)

w = occ.Wire([arc1, arc2, arc3, arc4])
f = occ.Face(w)
f.maxh = maxh
f.edges.maxh = 0.04
f.edges[0].name = 'force_bnd'
f.edges[1].name = 'free_bnd'
f.edges[2].name = 'pipette_bnd'
f.edges[3].name = 'free_bnd'

geo = occ.OCCGeometry(f, dim = 2)

ngmesh = geo.GenerateMesh(maxh=maxh,
    uselocalh=True,
    optsteps2d=3 )
mesh = Mesh(ngmesh)

###################  Solver  ##################################
t = Parameter(0)
dt = Parameter(2e-3)
model = CosmosModel(name='model', parentmesh=mesh, t0 = 0, t1 = 10,
                    dt = dt, t = t,
                    coupling_type = 'implicit', redistribute = True, iterate_ale = True,
                    root = '.', sample_rate = 100)

################### GRADIENT FLOW  ##################################
comp1 = model.create_compartment(name = 'comp1', boundary = 'free_bnd|force_bnd', bboundary = 'default', clamped_bbnd = "default")
pde1 = model.create_pde(name = 'geom_flow', pde_model=GeometricalFlowModel, compartment=comp1)

###################  VOLUME ADR  ##################################
comp2 = model.create_compartment(name = 'comp2', material = 'default', boundary = 'free_bnd|force_bnd|pipette_bnd')
pde2 = model.create_pde(name = 'concentration', pde_model=ADRVolumeStabBDF1Model, compartment=comp2)
ns = specialcf.normal(2)
pde2.set_params(
    Neu_bnd = 'force_bnd',
    d = 0.1,
    c = 1,
    gradu_bnd = ns,
    b = model.ale.wind,
    ale_velocity = model.ale.material_velocity
)

drag_coef = 1
bending_modulus = 0.01
species_const = 1
pde1.set_params(
    rhs = lambda: species_const*pde2.gfu/drag_coef*IfPos(x, 1, 0),
    alpha = bending_modulus/drag_coef
)

###################  ALE  ##################################
ale1 = model.create_ale('ale1', compartment=comp1)
ale1.set_normal_velocity(lambda: pde1.V_h)
ale1.set_tangential_velocity(lambda: CF((0,0)))

model.set_params(
    output_callables = {'energy': lambda: Integrate(0.5*(pde1.kappa_h - pde1.params['sp_curv'])**2, mesh, VOL_or_BND = BND),
                        'area': lambda: Integrate(1, mesh, VOL_or_BND = BND),
                        'volume': lambda: Integrate(1, mesh, VOL_or_BND = VOL)}
)

scene = Draw(pde2.gfu ,mesh)
for _ in model():
    scene.Redraw()
# %%


from ngsolve import *
from ngsolve.webgui import Draw
import netgen.occ as occ
from cosmos import *
import numpy as np
import logging
import pytest
from cosmos.core.model import CosmosModel
from cosmos.pde.willmore.geometrical_flow_model import GeometricalFlowModel
from cosmos.pde.adr.volume.adr_volume_bdf1_model import ADRVolumeBDF1Model
from cosmos.pde.adr.volume.adr_volume_stab_bdf1_model import ADRVolumeStabBDF1Model

logging.getLogger().setLevel(logging.INFO)


R = 1
maxh = 0.2

angle1_deg = 20
angle1 = angle1_deg/180*pi

angle2_deg = 45
angle2 = angle2_deg/180*pi

pnt0 = occ.Pnt(R*cos(angle1), -R*sin(angle1), 0)
pnt1 = occ.Pnt(R, 0, 0)
pnt2 = occ.Pnt(R*cos(angle1), R*sin(angle1), 0)
pnt3 = occ.Pnt(0, R, 0)
pnt4 = occ.Pnt(-R*cos(angle2), R*sin(angle2), 0)
pnt5 = occ.Pnt(-R, 0, 0)
pnt6 = occ.Pnt(-R*cos(angle2), -R*sin(angle2), 0)
pnt7 = occ.Pnt(0, -R, 0)


arc1 = occ.ArcOfCircle(pnt0, pnt1, pnt2)
arc2 = occ.ArcOfCircle(pnt2, pnt3, pnt4)
arc3 = occ.ArcOfCircle(pnt4, pnt5, pnt6)
arc4 = occ.ArcOfCircle(pnt6, pnt7, pnt0)

w = occ.Wire([arc1, arc2, arc3, arc4])
f = occ.Face(w)
f.maxh = maxh
f.edges.maxh = 0.04
f.edges[0].name = 'force_bnd'
f.edges[1].name = 'free_bnd'
f.edges[2].name = 'pipette_bnd'
f.edges[3].name = 'free_bnd'

geo = occ.OCCGeometry(f, dim = 2)

ngmesh = geo.GenerateMesh(maxh=maxh,
    uselocalh=True,
    optsteps2d=3 )
mesh = Mesh(ngmesh)

###################  Solver  ##################################
t = Parameter(0)
dt = Parameter(2e-3)
model = CosmosModel(name='model', parentmesh=mesh, t0 = 0, t1 = 10,
                    dt = dt, t = t,
                    coupling_type = 'implicit', redistribute = True,
                    root = '.', sample_rate = 100)

################### GRADIENT FLOW  ##################################
comp1 = model.create_compartment(name = 'comp1', boundary = 'free_bnd|force_bnd', bboundary = 'default', clamped_bbnd = "default")
pde1 = model.create_pde(name = 'geom_flow', pde_model=GeometricalFlowModel, compartment=comp1)

###################  VOLUME ADR  ##################################
comp2 = model.create_compartment(name = 'comp2', material = 'default', boundary = 'free_bnd|force_bnd|pipette_bnd')
pde2 = model.create_pde(name = 'concentration', pde_model=ADRVolumeBDF1Model, compartment=comp2)
ns = specialcf.normal(2)
pde2.set_params(
    Neu_bnd = 'force_bnd',
    d = 0.1,
    c = 1,
    gradu_bnd = ns,
    b = model.ale.wind
)

################### ALE  ##################################
drag_coef = 1
species_const = 1
bending_modulus = 0
ale1 = model.create_ale('ale1', compartment=comp1)
ale1.set_normal_velocity(lambda: (species_const*pde2.gfu/drag_coef + bending_modulus*pde1.V_h/drag_coef))
ale1.set_tangential_velocity(CF((0,0)))

model.set_params(
    output_callables = {'energy': lambda: Integrate(0.5*(pde1.kappa_h - pde1.params['sp_curv'])**2, mesh, VOL_or_BND = BND),
                        'area': lambda: Integrate(1, mesh, VOL_or_BND = BND),
                        'volume': lambda: Integrate(1, mesh, VOL_or_BND = VOL)}
)

scene = Draw(pde2.gfu ,mesh)
for _ in model():
    scene.Redraw()
