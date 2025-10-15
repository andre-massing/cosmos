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
dt = 1e-6
t = Parameter(0)
T = 1*dt
n = 200
sample_rate = np.maximum(int(abs(T/dt/n)), 1)
dt = Parameter(dt)
# t = Parameter(-4*60)

folderpath = './quintana_2d'

solvermesh = SolverMesh(mesh)
solvertime = SolverTime(dt, t.Get(), T, t_coef=t)
solver = Solver(solvermesh, solvertime, printing=True)

###################  WILLMORE FLOW  ##################################
input_params = {
    'clamped_bnd': 'membrane_bnd',
    'clamped_conormal': CF((0, -1)),
    'autoupdate': True
}
willmore = WillmoreBoundaryBDF1Model(solver, 1, domain = 'membrane', input_params=input_params)
willmore.set_input_fields({
    'spontaneous_curvature': 0,
    'elasticity_modulus': 1,
})

###################  ALE  ##################################
ale = ALEModel(solver, 2)
ale.set_bnd_displacement(willmore.displacement, domain = 'membrane', clamped_bnd='membrane_bnd', 
                         redistribute=True)

###################  SIMULATION  ##################################
scene = Draw(mesh.deformation, mesh)
for i, sol in enumerate(solver()):
    scene.Redraw()
# %%

# %%
