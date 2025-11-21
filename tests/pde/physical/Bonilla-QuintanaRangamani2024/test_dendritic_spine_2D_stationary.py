# %%
from ngsolve import *
from ngsolve.webgui import Draw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cosmos import *
from test_dendritic_spine_geom import generate_synapse2d

mesh, _ = generate_synapse2d(maxh=0.02)
mip = mesh(0.3, 0.7)

###################  PARAMETERS  ##################################
# T = 4*60
dt = 1e-2
t = Parameter(0)
T = 1
n = 200
sample_rate = np.maximum(int(abs(T/dt/n)), 1)
dt = Parameter(dt)
# t = Parameter(-4*60)

folderpath = './quintana_2d'

from cosmos.core.model import CosmosModel

model = CosmosModel(name = 'willmore_Mayte', parentmesh=mesh, t = t, t0 = t.Get(),
                    t1 = T, dt = dt, root = '.', sample_rate=10)

###################  WILLMORE FLOW  ##################################
comp1 = model.create_compartment(name = 'membrane', boundary = 'membrane', bboundary = 'membrane_bnd',
                                 clamped_bbnd = 'membrane_bnd')

# from cosmos.pde.willmore.geometrical_flow_model import GeometricalFlowModel
# geoflow = model.create_pde(name = 'geometric_flow', pde_model=GeometricalFlowModel, compartment=comp1)

from cosmos.pde.willmore.geometrical_flow_stationary_model import GeometricalFlowStationaryModel
geoflow = model.create_pde(name = 'geometric_flow', pde_model=GeometricalFlowStationaryModel, compartment=comp1)


###################  ALE  ##################################
ale = model.create_ale(name = 'ale', compartment=comp1)
ale.set_normal_velocity(geoflow.V_h)
ale.set_tangential_velocity(CF((0,0)))

###################  SIMULATION  ##################################
scene = Draw(mesh.deformation, mesh)
for i, sol in enumerate(model()):
    scene.Redraw()
# %%

# %%
