# %%

from cosmos.utils.generate_meshes import generate_volume_circle
from ngsolve import *
from ngsolve.webgui import Draw

R = 1
mesh = generate_volume_circle(maxh = 0.1, R = R)
Area0 = Integrate(1, mesh, VOL_or_BND=BND)

Draw(mesh)

print(mesh.ne)
print(mesh.nv)

from cosmos import *

import numpy as np

import os

T = 60
dt = 0.02
n = 200
sample_rate = np.maximum(int(T/dt/n), 1)
a_12 = 1
a_21 = 1
kappa1 = 0.4
kappa2 = 0.4
m = 4
s = 8

def solve_lomakin(i, angle):

    # folderpath = './lomakin_2d_' + str(i)
    # os.makedirs(folderpath, exist_ok = True)

    ###################  Solver  ##################################
    solvermesh = SolverMesh(mesh)
    solvertime = SolverTime(dt=dt, initial_t=0, final_t=T)
    solver = Solver(solvermesh, solvertime, iter = False, printing=False)


    ###################  SURFACE REACTIONS  ##################################
    u = ADRBoundaryBDF1Model(solver, 1, input_params={'u0': 0.5}, name = 'u')
    
    w0 = IfPos(x/R - angle, 0.1, 0.5)
    w = ADRBoundaryBDF1Model(solver, 2, input_params={'u0': 0.5}, name = 'w')

    ###################  ALE  ##################################
    ale = ALEModel(solver, 3)
    def V():
        Area = Integrate(1, mesh, VOL_or_BND=BND)
        stretch = (Area**(s+1))/(Area**s + Area0**s)
        term1 = u.sol**m/(u.sol**m + stretch**m)
        term2 = w.sol**m/(w.sol**m + 1)
        return term1 - term2
    ns = specialcf.normal(mesh.dim)
    def displ():
        return V()*dt*ns
    ale.set_bnd_displacement(displ, 'boundary', redistribute=True)

    def u_rhs():
        term1 = kappa1*V()*u.sol - u.sol**2 - a_12*u.sol*w.sol
        return term1
    u.set_input_fields({
        'd': 0.1,
        'c': -1,
        'b': ale.wind,
        'rhs': u_rhs
    })
    
    def w_rhs():
        term1 = -kappa2*V()*w.sol - w.sol**2 - a_21*u.sol*w.sol
        return term1
    w.set_input_fields({
        'd': 0.1,
        'c': -1,
        'rhs': w_rhs,
        'b': ale.wind
    })

    depletion = False
    scene = Draw(x, mesh, min = 0, max = 3)
    for i, sol in enumerate(solver()):
        if solver.current_time>10 and depletion == False:
            gfu = GridFunction(w.gfu_old.space)
            gfu.vec.data = w.gfu.vec.data
            gfu.Set(IfPos(x/(sqrt(x**2+y**2))-angle, 0.5*w.gfu, w.gfu), definedon = mesh.Boundaries('.*'))
            w.gfu_old.vec.data = gfu.vec.data
            w.gfu.vec.data = gfu.vec.data
            depletion = True
        scene.Redraw()
        

angles = [sqrt(3)/2, 1/2, 0, -1/2, -sqrt(3)/2]
for i, angle in enumerate(angles):
    solve_lomakin(i, angle)
# %%
