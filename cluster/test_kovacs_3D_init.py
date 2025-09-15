# %%

from ngsolve import *
from cosmos import *
from ngsolve.webgui import Draw
from cosmos.utils.generate_surface_meshes import generate_sphere
import numpy as np
np.random.seed(42)
from test_kovacs_config import CFG

import logging
logging.getLogger().setLevel(logging.DEBUG)

import os
filename = os.path.splitext(os.path.basename(__file__))[0]
os.makedirs(CFG.output_folder  + filename, exist_ok=True)

mesh, _ = generate_sphere(R=1, maxh = CFG.maxh)

dt = CFG.dt
Tend = CFG.Tend
solvertime = SolverTime(dt = dt, initial_t=0, final_t=5)

solvermesh = SolverMesh(mesh)
solver0 = Solver(solvermesh, solvertime, iter=True,
                name = filename)
solver0.output_params(CFG.output_folder, sample_rate=50)

u0 = ADRBoundaryBDF1Model(solver0, 1, name = 'u0_adr_boundary_bdf1', input_params = {'fes_order': 1})
w0 = ADRBoundaryBDF1Model(solver0, 2, name = 'w0_adr_boundary_bdf1', input_params = {'fes_order': 1})
n = len(u0.sol.vec.data)
epsilon1 = np.random.uniform(0, 0.01, n)
epsilon2 = np.random.uniform(0, 0.01, n)
u0.sol.vec.data = CFG.a+CFG.b+epsilon1
w0.sol.vec.data = CFG.b/(CFG.a+CFG.b)**2 + epsilon2

def u0_rhs():
    term1 = CFG.gamma*(CFG.a-u0.sol+u0.sol**2*w0.sol)
    return term1
u0.set_input_fields({
    "d": 1,
    "rhs": u0_rhs
    })
def w0_rhs():
    term1 = CFG.gamma*(CFG.b-u0.sol**2*w0.sol)
    return term1
w0.set_input_fields({
    "d": 10,
    "rhs": w0_rhs
    })

solver0.save_model_solution("u0_adr_boundary_bdf1", "sol")
solver0.save_model_solution("w0_adr_boundary_bdf1", "sol")

scene = Draw(mesh.deformation, mesh)
gfu = GridFunction(H1(mesh))
gfu.Set(u0.sol, definedon = mesh.Boundaries('.*'))
scene1=Draw(gfu)
try:
    for _ in solver0():
        scene.Redraw()
        gfu.Set(u0.sol, definedon = mesh.Boundaries('.*'))
        scene1.Redraw()
    np.save('test_kovacs_3D_u0_vec', u0.sol.vec.FV().NumPy())
    np.save('test_kovacs_3D_w0_vec', w0.sol.vec.FV().NumPy())
except Exception as e:
    with open(CFG.output_folder  + filename + '/error_file0.txt', "w") as f:
        f.write('Simulation terminated with error\n')
        f.write('Time: ' +  str(solver0.time.t.Get()) +', iter: '+ str(solver0.time.iter) + '\n')
        f.write('Cause: ' + str(e))