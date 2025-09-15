# %%

from ngsolve import *
from cosmos import *
from cosmos.utils.generate_surface_meshes import generate_sphere


ch_model = CahnHilliardBoundaryBachiniBDF1Model
M = 0.01
epsilon = 0.1
sigma = 1.5*sqrt(2)
ch_input_params = {
    "epsilon": epsilon,
    "sigma": sigma,
    "M": M,
    "fes_order": 1,
    "bounds": [-1, 1], 
    "mass_preserving": True
}

wm_model = WillmoreBoundaryInexBDF1Model
wm_input_params = {
    "autoupdate": True
}

maxh = 0.08
mesh, _ = generate_sphere(R=1, maxh = maxh)
dt = 1e-3
Tend = 1e-3
solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
solvermesh = SolverMesh(mesh)
solver = Solver(solvermesh, solvertime, iter=False)

willmore = wm_model(solver, 1, input_params = wm_input_params)
ale = ALEModel(solver, 2)
ch = ch_model(solver, 2, input_params=ch_input_params)

ch.phase.Set(sinh(x/sqrt(2))/cosh(x/sqrt(2)), definedon = mesh.Boundaries('.*'), dual = True)

for _ in solver():
    pass

print('Cahn-Hilliard + Willmore flow is ok')

from ngsolve import *
from cosmos import *
from ngsolve.webgui import Draw
from cosmos.utils.generate_surface_meshes import generate_sphere
import numpy as np
np.random.seed(42)

mesh, _ = generate_sphere(R=1, maxh = 0.1)
solvermesh = SolverMesh(mesh)
solvertime = SolverTime(dt = 1e-3, initial_t=0, final_t=1e-3)
solver = Solver(solvermesh, solvertime, iter=True)

mc = MeanCurvatureBoundaryBDF1Model(solver, 1, name = 'mc_boundary_bdf1', input_params={'kappa': 0.01})
u = ADRBoundaryBDF1Model(solver, 2, name = 'u_adr_boundary_bdf1')
w = ADRBoundaryBDF1Model(solver, 3, name = 'w_adr_boundary_bdf1')
ale = ALEModel(solver, 3, name = 'ale_model')

ale.set_bnd_displacement(mc.displacement, 'default', redistribute=False)

u.set_input_fields({
    "d": 1,
    })
w.set_input_fields({
    "d": 10,
    })

for _ in solver():
    pass

print('ADR + Mean Curvature flow is ok')