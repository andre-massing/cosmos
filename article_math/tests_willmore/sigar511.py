from ngsolve import *
from cosmos.utils.generate_surface_meshes import generate_sigar
import os
import numpy as np
import pandas as pd
from cosmos.solvers.solvers import Dynamic
from ngsolve.webgui import Draw
from myngspy import *
from cosmos.pdes.pde_tools import SaveSolution

testname = 'sigar511'

def test_sigar511(results_path, solver_name, pde_constructor, **init_kwargs):

    folderpath = os.path.join(results_path, solver_name, testname)

    mesh, _ = generate_sigar(maxh = 0.25, r=1, h=5)

    dt = 1e-3
    T = 0.3
    n = 200
    sample_rate = np.maximum(int(T/dt/n), 1)

    os.makedirs(folderpath, exist_ok=True)

    t = Parameter(0.0)
    dt = Parameter(dt)

    # With boundary conditions
    willmore = pde_constructor(name = [solver_name + 'dX', solver_name + 'mc'], sp_curv=-3, **init_kwargs)
    sol_save = SaveSolution(folderpath = folderpath,
                            filename = 'sigar511',
                            sample_rate = sample_rate)
    willmore.SaveSol(sol_save)

    solver = Dynamic(mesh = mesh, dt=dt, T=T, t=t)
    solver.AddPDE(willmore)
    solver.ale.deformation_field = willmore.gfu.components[0]

    energy = []
    time = []
    scene = Draw(willmore.mean_curvature, mesh, deformation = willmore.displacement_tot)
    try:
        for sol in solver():
            scene.Redraw()
            energy_i = willmore.GetEnergy(solver)
            energy.append(energy_i)
            time.append(t.Get())
    except Exception as e: 
        print(e)
        print('Simulation terminated with error')

    energy = np.array(energy)
    time = np.array(time)
    if len(time)>n:
        indices = np.linspace(0, len(time) - 1, n, dtype=int)
        energy = energy[indices]
        time = time[indices]

    df = pd.DataFrame(np.column_stack([time, energy]), columns = ['Time',  'Energy'])
    df.to_csv(os.path.join(folderpath,  'Energy' + solver_name + '.dat'), sep='\t', index=False)