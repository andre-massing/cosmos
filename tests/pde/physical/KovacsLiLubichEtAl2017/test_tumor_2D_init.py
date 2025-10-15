# %%

from ngsolve import *
from ngsolve.webgui import Draw
from cosmos import *
from cosmos.utils.generate_meshes import generate_volume_circle
import numpy as np

params = {}
params['a'] = 0.1
params['b'] = 0.9
params['gamma'] = 100
params['delta'] = 0.4
params['dt'] = 1e-3
params['Tend'] = 5
params['maxh'] = 0.07

Id = CF((x, 0))
n = specialcf.normal(2)

mesh = generate_volume_circle(maxh  = params['maxh'])
solvermesh = SolverMesh(mesh)
dt = params['dt']
Tend = params['Tend']
solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
solver = Solver(solvermesh, solvertime, iter=True,
                printing=True)

u = ADRVolumeBDF1Model(solver, 1, name = 'u_adr_boundary_bdf1')
w = ADRVolumeBDF1Model(solver, 2, name = 'w_adr_boundary_bdf1')

def u_rhs():
    term1 = params['gamma']*(params['a']-u.sol+u.sol**2*w.sol)
    return term1
u.set_input_fields({
    "d": 1,
    "rhs": u_rhs,
    })
def w_rhs():
    term1 = params['gamma']*(params['b']-u.sol**2*w.sol)
    return term1
w.set_input_fields({
    "d": 10,
    "rhs": w_rhs,
    })
def mc_rhs():
    ns = specialcf.normal(mesh.dim)
    term1 = params['delta']*u.sol*ns
    return term1

n = len(u.sol.vec.data)
np.random.seed(42)
epsilon1 = np.random.uniform(0, 0.01, n)
epsilon2 = np.random.uniform(0, 0.01, n)
u.sol.vec.data = params['a']+params['b']+epsilon1
w.sol.vec.data = params['b']/(params['a']+params['b'])**2 + epsilon2

scene = Draw(u.sol, mesh)
for _ in solver():
    scene.Redraw()

np.save('test_kovacs_2D_u0_vec', u.sol.vec.FV().NumPy())
np.save('test_kovacs_2D_w0_vec', w.sol.vec.FV().NumPy())