from ngsolve import *
from cosmos.pdes.pde_tools import gradient
from cosmos.utils.generate_surface_meshes import generate_half_sphere
import os
import numpy as np
import pandas as pd
from cosmos.solvers.solvers import Dynamic
from cosmos.pdes.pde_tools import SaveSolution
from ngsolve.webgui import Draw
from myngspy import *

testname = 'illposed_case'

def test_illposed(results_path, solver_name, pde_constructor, **init_kwargs):

    folderpath = os.path.join(results_path, solver_name, testname)

    dt = 0.01
    T = 1
    n = 200
    mesh, _ = generate_half_sphere(maxh = 0.1)
    sample_rate = np.maximum(int(T/dt/n), 1)

    dt = Parameter(0.01)
    t = Parameter(0.0)

    mass = []
    time = []

    def AandB(t):
        A = CF((cos(t), -sin(t), 0,\
                        sin(t), cos(t), 0,\
                            0, 0, 1), dims = (3,3))
        B = CF((0, 0, 0))
        return A, B

    A, B = AandB(t)
    detJ = Det(A)
    invA = Cof(A).trans/detJ
    inv_phi = invA*(CF((x,y,z)) - B) # Inverse of deformation map
    w_phi = A.Diff(t)*inv_phi + B.Diff(t)
    n_ex = Cof(A)*inv_phi/Norm(Cof(A)*inv_phi)
    P_ex = Id(3) - OuterProduct(n_ex, n_ex)
    A1, B1 = AandB(t+dt)
    dX_ex = (A1-A)*inv_phi+(B1-B)

    u_ex = exp(-3*(x**2 + y**2))
    b = CF((z,0,-x))*(1-exp(-10*z))
    neu_b = {'bottom': CF((0, 0, 0))}

    pde = pde_constructor(b = b, neu_b = neu_b,
        u0=u_ex, name = solver_name, **init_kwargs)
    sol_save = SaveSolution(folderpath = folderpath,
                            filename = solver_name,
                            sample_rate = sample_rate)
    pde.SaveSol(sol_save)

    solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)
    solver.AddPDE(pde)
    solver.ale.deformation_field = dX_ex

    scene =Draw(pde.solute, mesh, deformation = solver.ale.deformation)
    for sol in solver():
        scene.Redraw()
        mass_i = pde.GetTotMass(solver)
        mass.append(mass_i)
        time.append(t.Get())

    mass = np.array(mass)
    mass = (mass - mass[0])/mass[0]

    mass = np.array(mass)
    time = np.array(time)
    if len(time)>n:
        indices = np.linspace(0, len(time) - 1, n, dtype=int)
        mass = mass[indices]
        time = time[indices]

    df = pd.DataFrame(np.column_stack([time, mass]), columns = ['Time',  'mass'])
    df.to_csv(os.path.join(folderpath, 'mass' + solver_name + '.dat'), sep='\t', index=False)