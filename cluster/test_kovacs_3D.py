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

mesh, _ = generate_sphere(R=1, maxh = CFG.maxh)

Id = CF((x, 0, 0))
n = specialcf.normal(3)

solvermesh = SolverMesh(mesh)
dt = CFG.dt
Tend = CFG.Tend
solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
solver = Solver(solvermesh, solvertime, iter=True,
                name = filename)
solver.output_params(CFG.output_folder, sample_rate=20)

mc = MeanCurvatureBoundaryBDF1Model(solver, 1, name = 'mc_boundary_bdf1', input_params={'kappa': 0.01})
u = ADRBoundaryBDF1Model(solver, 2, name = 'u_adr_boundary_bdf1')
w = ADRBoundaryBDF1Model(solver, 3, name = 'w_adr_boundary_bdf1')
ale = ALEModel(solver, 3, name = 'ale_model')
try:
    u0 = np.load('test_kovacs_3D_u0_vec.npy')
    w0 = np.load('test_kovacs_3D_w0_vec.npy')
    u.sol.vec.data = u0
    w.sol.vec.data = w0
except:
    import test_kovacs_3D_init
    u0 = np.load('test_kovacs_3D_u0_vec.npy')
    w0 = np.load('test_kovacs_3D_w0_vec.npy')
    u.sol.vec.data = u0
    w.sol.vec.data = w0

ale.set_bnd_displacement(mc.displacement, 'default', redistribute=False)
def u_rhs():
    term1 = CFG.gamma*(CFG.a-u.sol+u.sol**2*w.sol)
    return term1
u.set_input_fields({
    "d": 1,
    "rhs": u_rhs,
    })
def w_rhs():
    term1 = CFG.gamma*(CFG.b-u.sol**2*w.sol)
    return term1
w.set_input_fields({
    "d": 10,
    "rhs": w_rhs,
    })
def mc_rhs():
    ns = specialcf.normal(mesh.dim)
    term1 = CFG.delta*u.sol*ns
    return term1
mc.set_input_fields({"rhs": mc_rhs})

solver.save_model_solution("ale_model", "displacement")
solver.save_model_solution("u_adr_boundary_bdf1", "sol")
solver.save_model_solution("w_adr_boundary_bdf1", "sol")

scene = Draw(mesh.deformation, mesh)
gfu = GridFunction(H1(mesh))
gfu.Set(u.sol, definedon = mesh.Boundaries('.*'))
scene1=Draw(gfu)
try:
    with open(CFG.output_folder + filename + '/simulation.txt', "w") as f:
        f.write('Area\tVolume\n')
        for _ in solver():
            scene.Redraw()
            gfu.Set(u.sol, definedon = mesh.Boundaries('.*'))
            scene1.Redraw()
            area = Integrate(1, mesh, VOL_or_BND = BND)
            volume = Integrate(Id*n, mesh, VOL_or_BND = BND)
            f.write(str(area) + '\t' + str(volume) + '\n')
except Exception as e:
    with open(CFG.output_folder + filename + '/error_file.txt', "w") as f:
        f.write('Simulation terminated with error\n')
        f.write('Time: ' +  str(solver.time.t.Get()) +', iter: '+ str(solver.time.iter) + '\n')
        f.write('Cause: ' + str(e))

# %%
