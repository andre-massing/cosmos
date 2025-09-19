from ngsolve import *
from cosmos import *
from ngsolve.webgui import Draw
from cosmos.utils.generate_meshes import generate_boundary_torus
from scipy.integrate import solve_ivp
import numpy as np
import logging
import pytest
import pandas as pd

logging.getLogger().setLevel(logging.INFO)

@pytest.mark.parametrize("maxh", [0.2, 0.1])
@pytest.mark.parametrize("redistribute", [None, 'DuanLi', 'MDR'])
def test_torus(
        request,
        artifacts_path,
        redistribute, 
        maxh):
    
    out = artifacts_path
    filename = request.function.__name__

    Tend = 2

    dt = 1e-3
    n = 200
    mesh = generate_boundary_torus(maxh = maxh, r=1, R=2)

    solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
    solvermesh = SolverMesh(mesh)
    solver = Solver(solvermesh, solvertime)
    solver.output_params(out, sample_rate=50)
    
    ale = ALEModel(solver, 2)
    if redistribute:
        pde = WillmoreBoundaryBDF1Model(solver, 1, input_params={'autoupdate': True})
        ale.set_bnd_displacement(pde.displacement, domain='default', 
                                    redistribute=True, redistribute_type=redistribute)
    else:
        pde = WillmoreBoundaryBDF1Model(solver, 1)
        ale.set_bnd_displacement(pde.displacement, domain='default')

    solver.save_model_solution(pde.name, 'displacement')
    solver.save_model_solution(pde.name, 'mean_curvature')

    energy = []
    time = []
    for _ in solver():
        energy_i = pde.energy
        energy.append(energy_i)
        time.append(solver.current_time)

    os.makedirs(out, exist_ok=True)

    energy = np.array(energy)
    time = np.array(time)
    if len(time)>n:
        indices = np.linspace(0, len(time) - 1, n, dtype=int)
        energy = energy[indices]
        time = time[indices]

    df = pd.DataFrame(np.column_stack([time, energy]), columns = ['Time',  'energy'])
    df.to_csv(os.path.join(out, 'energy_' + pde.name + '.dat'), sep='\t', index=False)

    assert 1