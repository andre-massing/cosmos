import pytest
from ngsolve import *
from cosmos import *
from cosmos.utils.generate_meshes import generate_boundary_ellipse

@pytest.fixture
def time_params():
    time_params = {}
    time_params['dt'] = 1e-2
    time_params['final_t'] = 1.5
    time_params['initial_t'] = 0
    return time_params

@pytest.fixture
def maxh():
    return 0.4 

@pytest.mark.parametrize('redistribute, redistribute_type', [(False, 'DuanLi'), (True, 'DuanLi')])
def test_helfrich_blood_cell_441(
        request,
        artifacts_path,
        time_params,
        maxh,
        redistribute,
        redistribute_type):
    
    out = artifacts_path
    filename = request.function.__name__
    
    solvertime = SolverTime(**time_params)
    mesh = generate_boundary_ellipse(maxh=maxh, a=4, b=4, c=1)
    solvermesh = SolverMesh(mesh)
    solver = Solver(solvermesh, solvertime, iter=False,
                    name = filename, printing=True)
    solver.output_params(out, sample_rate=2)

    willmore = WillmoreBoundaryAPVPBDF1Model(solver, 1, input_params = {"autoupdate": True})

    ale = ALEModel(solver, 2)
    ale.set_bnd_displacement(willmore.displacement, 'default', redistribute=redistribute,
                             redistribute_type=redistribute_type)

    solver.save_model_solution(willmore)
    f = open(os.path.join(out, filename, 'simulation.txt'), "w")

    f.write('Time\tArea\tVolume\tEnergy\n')
    Id = CF((x, 0, 0))
    n = specialcf.normal(3)
    for _ in solver():
        area = Integrate(1, mesh, VOL_or_BND = BND)
        volume = Integrate(Id*n, mesh, VOL_or_BND = BND)
        energy = willmore.energy
        f.write(str(solver.time.t.Get()) + '\t' + str(area) + '\t' + str(volume) + '\t' + str(energy) +'\n')

@pytest.mark.parametrize('redistribute, redistribute_type', [(False, 'DuanLi'), (True, 'DuanLi')])
def test_helfrich_blood_cell_551(
        request,
        artifacts_path,
        time_params,
        maxh,
        redistribute,
        redistribute_type):
    
    out = artifacts_path
    filename = request.function.__name__
    
    solvertime = SolverTime(**time_params)
    mesh = generate_boundary_ellipse(maxh=maxh, a=5, b=5, c=1)
    solvermesh = SolverMesh(mesh)
    solver = Solver(solvermesh, solvertime, iter=False,
                    name = filename, printing=True)
    solver.output_params(out, sample_rate=2)

    willmore = WillmoreBoundaryAPVPBDF1Model(solver, 1, input_params = {"autoupdate": True})

    ale = ALEModel(solver, 2)
    ale.set_bnd_displacement(willmore.displacement, 'default', redistribute=redistribute,
                             redistribute_type=redistribute_type)

    solver.save_model_solution(willmore)
    f = open(os.path.join(out, filename, 'simulation.txt'), "w")

    f.write('Time\tArea\tVolume\tEnergy\n')
    Id = CF((x, 0, 0))
    n = specialcf.normal(3)
    for _ in solver():
        area = Integrate(1, mesh, VOL_or_BND = BND)
        volume = Integrate(Id*n, mesh, VOL_or_BND = BND)
        energy = willmore.energy
        f.write(str(solver.time.t.Get()) + '\t' + str(area) + '\t' + str(volume) + '\t' + str(energy) +'\n')