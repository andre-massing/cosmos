import pytest
from ngsolve import *
from cosmos import *
from cosmos.utils.generate_meshes import generate_boundary_ellipse

@pytest.fixture
def time_params():
    time_params = {}
    time_params['dt'] = 1e-4
    time_params['final_t'] = 5
    time_params['initial_t'] = 0
    return time_params

@pytest.fixture
def ch_params(ch_params_type):
    params = {}
    params['M'] = 0.001
    params['sigma'] = 1.5*sqrt(2)
    if ch_params_type == 1:
        params['fes_order'] = 1
    if ch_params_type == 2:
        params['fes_order'] = 1
        params['bounds'] = [-1, 1]
        params['mass_preserving'] = True
    if ch_params_type == 3:
        params['fes_order'] = 2
    return params

@pytest.fixture
def solver_time_params():
    params = {}
    params['dt'] = 1e-3
    params['initial_t'] = 0
    params['final_t'] = 5
    return params

@pytest.mark.parametrize("ch_solver, ch_params_type", 
    [(CahnHilliardBoundaryBachiniBDF1Model, 1),
     (CahnHilliardBoundaryBachiniBDF1Model, 2),
     (CahnHilliardBoundaryBachiniBDF1Model, 3)])
def test_cahn_hilliard_spine(
        request,
        cosmos_root,
        artifacts_path,
        time_params,
        ch_solver,
        ch_params):
    
    out = artifacts_path
    filename = request.function.__name__

    sample_rate = 50
    
    solvertime = SolverTime(**time_params)
    mesh = Mesh(os.path.join(cosmos_root, 'data/geometries',  'spine_fine.vol'))
    solvermesh = SolverMesh(mesh)
    solver = Solver(solvermesh, solvertime, iter=False,
                    name = filename, printing=True)
    solver.output_params(out, sample_rate=50)

    ch = ch_solver(solver, 3, input_params=ch_params)

    f = open(os.path.join(out, filename, 'simulation.txt'), "w")

    def timestep_f(t):
        return 1e-2 - (1e-2 - time_params['dt'])*exp(-50*t) 
    
    solver.save_model_solution(ch.name, "phase")
    solver.save_model_solution(ch.name, "potential")

    f_ch = open(os.path.join(out, filename, ch.name, 'ch_simulation.txt'), "w")
    f_ch.write('Time\tMass\tEnergy\tMaxValue\tMinValue\n')
    for _ in solver():
        solver.time.input_params["dt"] = timestep_f(solver.time.t.Get())
        if solver.time.iter%sample_rate == 0:
            ch_mass = Integrate(ch.phase, mesh, VOL_or_BND = BND)
            ch_energy = ch.energy
            ch_minvalue = ch.phase.vec.FV().NumPy().min()
            ch_maxvalue = ch.phase.vec.FV().NumPy().max()
            f_ch.write(str(solver.time.t.Get()) + '\t' + str(ch_mass) + '\t' + str(ch_energy) 
                       + '\t' + str(ch_maxvalue) + '\t' + str(ch_minvalue) + '\n')