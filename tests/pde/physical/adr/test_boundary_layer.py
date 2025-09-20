from ngsolve import *
from cosmos import *
from ngsolve.webgui import Draw
from cosmos.utils.generate_meshes import generate_boundary_half_sphere
from cosmos.utils.tools import gradient
import numpy as np
import logging
import pytest
import pandas as pd

logging.getLogger().setLevel(logging.INFO)

@pytest.fixture
def u0():
    return exp(-3*(x**2 + y**2))

@pytest.fixture
def input_params(params_type, u0):
    if params_type == 0:
        params = {
            "u0": u0,
        }
    elif params_type == 1:
        params = {
            "u0": u0,
            "mass_preserving": True,
        }
    elif params_type == 2:
        params = {
            "u0": u0,
            "bounds": [-1, 1],
        }
    elif params_type == 3:
        params = {
            "u0": u0,
            "mass_preserving": True,
            "bounds": [-1, 1],
        }
    params['Neu_bnd'] = 'bboundary'
    return params

@pytest.mark.parametrize("adr_solver", [
    ADRBoundaryBDF1Model,
    ADRBoundaryBDF2Model,
    ADRBoundaryStabBDF1Model,
    ADRBoundaryStabBDF2Model
])
@pytest.mark.parametrize("params_type", [0, 1, 2, 3])
def test_boundary_layer(
        request,
        artifacts_path,
        adr_solver,
        input_params):
    
    out = artifacts_path
    filename = request.function.__name__

    dt = 0.01
    T = 1
    n = 200
    mesh = generate_boundary_half_sphere(maxh = 0.1)
    sample_rate = np.maximum(int(T/dt/n), 1)

    t = Parameter(0.0)
    solvertime = SolverTime(dt = dt, initial_t=0, final_t=T, t_coef=t)
    solvermesh = SolverMesh(mesh)
    solver = Solver(solvermesh, solvertime)
    solver.output_params(out, sample_rate)

    mass = []
    time = []

    def A_B(t):
        A = CF((cos(t), -sin(t), 0,\
                    sin(t), cos(t), 0,\
                        0, 0, 1), dims = (3,3))
        B = CF((0, 0, 0))
        return A, B
    A, B = A_B(t)
    detJ = Det(A)
    invA = Cof(A).trans/detJ
    inv_phi = invA*(CF((x,y,z)) - B) # Inverse of deformation map

    Ap, Bp = A_B(t+dt)
    displacement_ex = (Ap-A)*inv_phi+(Bp-B)
    ale = ALEModel(solver=solver, model_order=1, name='displacement')
    ale.set_bnd_displacement(displacement_ex, 'default')

    b = CF((z,0,-x))*(1-exp(-10*z))

    pde = adr_solver(solver, 2, input_params = input_params)
    pde.set_input_fields({
        "b": b,
        "u_bnd": CF(0),
        "gradu_bnd": CF((0, 0, 0))
    })

    solver.save_model_solution(pde)

    for sol in solver():
        mesh.SetDeformation(solver.mesh.curr_deformation)
        mass_i = Integrate(pde.sol, mesh, BND)
        mesh.SetDeformation(solver.mesh.prev_deformation[-1])
        mass.append(mass_i)
        time.append(t.Get())

    mass = np.array(mass)
    mass = np.abs((mass - mass[0])/mass[0])

    mass = np.array(mass)
    time = np.array(time)
    if len(time)>n:
        indices = np.linspace(0, len(time) - 1, n, dtype=int)
        mass = mass[indices]
        time = time[indices]

    os.makedirs(out, exist_ok=True)

    df = pd.DataFrame(np.column_stack([time, mass]), columns = ['Time',  'mass'])
    df.to_csv(os.path.join(out, 'mass_' + pde.name + '.dat'), sep='\t', index=False)
    
    assert 1