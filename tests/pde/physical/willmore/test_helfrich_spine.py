import pytest
from ngsolve import *
from cosmos import *
from cosmos.utils.generate_meshes import generate_boundary_ellipse

@pytest.fixture
def time_params():
    time_params = {}
    time_params['dt'] = 1e-6
    time_params['final_t'] = 0.05
    time_params['initial_t'] = 0
    return time_params

@pytest.mark.parametrize('redistribute, redistribute_type', [(True, 'DuanLi'), (True, 'MDR')])
@pytest.mark.parametrize('wm_solver_type', (WillmoreBoundaryVPBDF1Model, WillmoreBoundaryAPBDF1Model,
                                            WillmoreBoundaryAPVPBDF1Model, WillmoreBoundaryInexBDF1Model))
def test_helfrich_spine(
        request,
        cosmos_root,
        artifacts_path,
        time_params,
        redistribute,
        redistribute_type,
        wm_solver_type):
    
    out = artifacts_path
    filename = request.function.__name__
    
    solvertime = SolverTime(**time_params)
    mesh = Mesh(os.path.join(cosmos_root, 'data/geometries',  'spine_fine.vol'))
    solvermesh = SolverMesh(mesh)
    solver = Solver(solvermesh, solvertime, iter=False,
                    name = filename, printing=True)
    solver.output_params(out, sample_rate=10)

    willmore = wm_solver_type(solver, 1, input_params = {"autoupdate": True})

    ale = ALEModel(solver, 2)
    ale.set_bnd_displacement(willmore.displacement, 'default', redistribute=redistribute,
                             redistribute_type=redistribute_type)

    solver.save_model_solution(willmore)
    f = open(os.path.join(out, filename, 'simulation.txt'), "w")

    def timestep_f(t):
        return 1e-4 - (1e-4 - time_params['dt'])*exp(-50*t) 

    f.write('Time\tArea\tVolume\tEnergy\n')
    Id = CF((x, 0, 0))
    n = specialcf.normal(3)
    for _ in solver():
        solver.time.input_params["dt"] = timestep_f(solver.time.t.Get())
        area = Integrate(1, mesh, VOL_or_BND = BND)
        volume = Integrate(Id*n, mesh, VOL_or_BND = BND)
        energy = willmore.energy
        f.write(str(solver.time.t.Get()) + '\t' + str(area) + '\t' + str(volume) + '\t' + str(energy) +'\n')