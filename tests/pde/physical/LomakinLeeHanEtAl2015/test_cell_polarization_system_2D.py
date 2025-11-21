# %%

from ngsolve import *
from ngsolve.webgui import Draw
from cosmos import *
import netgen.occ as occ
import numpy as np
import logging
import pytest

logging.getLogger().setLevel(logging.INFO)

dt = 2e-2
t0 = 0
t1 = 50

a_12 = 1
a_21 = 1
kappa1 = 0.4
kappa2 = 0.4
m = 4
s = 8

angle_deg = 30

R = 1
face = occ.WorkPlane(occ.Axes((0,0,0), n=occ.Z, h=occ.X)).Circle(0, 0, R).Face()
face.edges.name = 'boundary'
face.edges[0].maxh = 0.04
geo = occ.OCCGeometry(face, dim = 2)
mesh = geo.GenerateMesh(maxh=1, optsteps2d=3)
mesh = Mesh(mesh)
Area0 = Integrate(1, mesh, VOL_or_BND=BND)

###################  Solver  ##################################
from cosmos.core.model import CosmosModel
model = CosmosModel(name = 'system_Lomakin', parentmesh = mesh, dt = dt, t0 = t0, t1 = t1,
                    root = '.', sample_rate = 10)

comp1 = model.create_compartment(name = 'compartment', boundary = 'boundary', bboundary = '')

from cosmos.pde.adr.boundary.adr_boundary_system_bdf1_model import ADRBoundarySystemBDF1Model
adr_system = model.create_pde(name = 'adr_system', pde_model=ADRBoundarySystemBDF1Model, compartment=comp1,
                              dim = 2)

ale = model.create_ale(name = 'ale', compartment=comp1)
def V():
    Area = Integrate(1, mesh, VOL_or_BND=BND)
    stretch = (Area**(s+1))/(Area**s + Area0**s)
    term1 = adr_system.sol[0]**m/(adr_system.sol[0]**m + stretch**m)
    term2 = adr_system.sol[1]**m/(adr_system.sol[1]**m + 1)
    return term1 - term2
ale.set_normal_velocity(V)
ale.set_tangential_velocity(CF((0,0)))

def u_rhs():
    term1 = kappa1*ale.gfu_norm_vel*adr_system.sol[0] - adr_system.sol[0]**2 - a_12*adr_system.sol[0]*adr_system.sol[1]
    return term1
def w_rhs():
        term1 = -kappa2*ale.gfu_norm_vel*adr_system.sol[1] - adr_system.sol[1]**2 - a_21*adr_system.sol[0]*adr_system.sol[1]
        return term1

ns = specialcf.normal(2)
adr_system.set_params(
    u0_1 = 0.5,
    d_1 = 0.1,
    c_1 = -1,
    b_1 = model.ale.wind,
    rhs_1 = u_rhs,
    u0_2 = 0.5,
    d_2 = 0.1,
    c_2 = -1,
    b_2 = model.ale.wind,
    rhs_2 = w_rhs,
    ale_velocity = ale.ale_vel
)

output_callables = {
    'mass': lambda: Integrate(adr_system.sol[0], mesh, VOL_or_BND = BND),
    'max_value': lambda: adr_system.sol[0].vec.FV().NumPy().max(),
    'min_value': lambda: adr_system.sol[0].vec.FV().NumPy().min()
}

model.set_params(
     output_callables = output_callables
)

depletion = False
scene = Draw(model.ale.gfu_bnd_ale, mesh)
for _ in model():
    if model.t.Get()>10 and depletion == False:
        gfu = GridFunction(adr_system.sol[1].space)
        gfu.vec.data = adr_system.sol[1].vec.data
        gfu.Set(IfPos(x/(sqrt(x**2+y**2))-cos(angle_deg/180*pi), 0.1*adr_system.sol[1], adr_system.sol[1]), definedon = mesh.Boundaries('.*'))
        adr_system.sol[1].vec.data = gfu.vec.data
        depletion = True
    scene.Redraw()
# %%

# %%
