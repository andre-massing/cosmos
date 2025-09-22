from ngsolve import *
from cosmos import *
from ngsolve.webgui import Draw
from cosmos.utils.generate_meshes import generate_boundary_sigar
from scipy.integrate import solve_ivp
import numpy as np
import logging
import pytest
import pandas as pd

logging.getLogger().setLevel(logging.INFO)

@pytest.fixture()
def Tend(shape):
    if shape == '31':
        return 1
    elif shape == '51':
        return 0.3

@pytest.mark.parametrize("maxh", [0.25, 0.125])
@pytest.mark.parametrize("redistribute", [None, 'DuanLi', 'MDR'])
@pytest.mark.parametrize("shape", ['31', '51'])
def test_sigar(
        request,
        artifacts_path,
        redistribute, 
        Tend,
        shape,
        maxh):
    
    out = artifacts_path
    filename = request.function.__name__

    dt = 1e-3
    n = 200
    if shape == '31':
        mesh = generate_boundary_sigar(maxh = maxh, r=1, h=3)
    elif shape == '51':
        mesh = generate_boundary_sigar(maxh = maxh, r=1, h=5)

    solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
    solvermesh = SolverMesh(mesh)
    solver = Solver(solvermesh, solvertime, printing=True)
    if shape == '31':
        solver.output_params(out, sample_rate=50)
    elif shape =='51':
        solver.output_params(out, sample_rate=20)
    
    ale = ALEModel(solver, 2)
    if redistribute:
        pde = WillmoreBoundaryBDF1Model(solver, 1, input_params={'autoupdate': True})
        ale.set_bnd_displacement(pde.displacement, domain='default', 
                                    redistribute=True, redistribute_type=redistribute)
    else:
        pde = WillmoreBoundaryBDF1Model(solver, 1)
        ale.set_bnd_displacement(pde.displacement, domain='default')
    if shape == '31':
        pde.set_input_fields({
            "spontaneous_curvature": -2
        })
    elif shape == '51':
        pde.set_input_fields({
            "spontaneous_curvature": -3
        })

    solver.save_model_solution(pde)

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