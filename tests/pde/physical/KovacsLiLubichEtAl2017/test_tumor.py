from ngsolve import *
from cosmos import *
from cosmos.utils.generate_meshes import generate_boundary_sphere
import numpy as np
import logging
import pytest

logging.getLogger().setLevel(logging.INFO)

@pytest.fixture
def params():
    params = {}
    params['a'] = 0.1
    params['b'] = 0.9
    params['gamma'] = 100
    params['delta'] = 0.4
    params['dt'] = 1e-3
    params['Tend'] = 3
    params['maxh'] = 0.07
    return params


@pytest.mark.parametrize("redistribute", [None, 'DuanLi', 'MDR'])
@pytest.mark.parametrize("adr_solver", [
    ADRBoundaryBDF1Model,
    ADRBoundaryStabBDF1Model
])
def test_tumor3D(
        request,
        artifacts_path,
        cosmos_root,
        adr_solver,
        redistribute,
        params):
    
    out = artifacts_path
    filename = request.function.__name__

    Id = CF((x, 0, 0))
    n = specialcf.normal(3)

    mesh = generate_boundary_sphere(maxh  = params['maxh'])
    solvermesh = SolverMesh(mesh)
    dt = params['dt']
    Tend = params['Tend']
    solvertime = SolverTime(dt = dt, initial_t=0, final_t=Tend)
    solver = Solver(solvermesh, solvertime, iter=True,
                    name = filename, printing=True)
    solver.output_params(out, sample_rate=30)

    mc = MeanCurvatureBoundaryBDF1Model(solver, 1, name = 'mc_boundary_bdf1', input_params={'kappa': 0.01})
    u = adr_solver(solver, 2, name = 'u_adr_boundary_bdf1')
    w = adr_solver(solver, 3, name = 'w_adr_boundary_bdf1')
    ale = ALEModel(solver, 3, name = 'ale_model')
    u0 = np.load(os.path.join(cosmos_root, 'tests/pde/physical/KovacsLiLubichEtAl2017/test_kovacs_3D_u0_vec.npy'), allow_pickle=False)
    w0 = np.load(os.path.join(cosmos_root, 'tests/pde/physical/KovacsLiLubichEtAl2017/test_kovacs_3D_w0_vec.npy'), allow_pickle=False)
    u.sol.vec.data = u0
    w.sol.vec.data = w0

    if redistribute:
        ale.set_bnd_displacement(mc.displacement, 'default',
                                 redistribute=True, redistribute_type=redistribute)
        def u_rhs():
            term1 = params['gamma']*(params['a']-u.sol+u.sol**2*w.sol)
            return term1
        u.set_input_fields({
            "b": ale.wind,
            "d": 1,
            "rhs": u_rhs,
            })
        def w_rhs():
            term1 = params['gamma']*(params['b']-u.sol**2*w.sol)
            return term1
        w.set_input_fields({
            "b": ale.wind,
            "d": 10,
            "rhs": w_rhs,
            })
        def mc_rhs():
            ns = specialcf.normal(mesh.dim)
            term1 = params['delta']*u.sol*ns
            return term1
        mc.set_input_fields({"rhs": mc_rhs})
    else:
        ale.set_bnd_displacement(mc.displacement, 'default')
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
        mc.set_input_fields({"rhs": mc_rhs})
    mc.set_input_fields({"rhs": mc_rhs})
    
    solver.save_model_solution(u)

    with open(os.path.join(out, 'simulation.txt'), "w") as f:
        f.write('Area\tVolume\n')
        for _ in solver():
            area = Integrate(1, mesh, VOL_or_BND = BND)
            volume = Integrate(Id*n, mesh, VOL_or_BND = BND)
            f.write(str(area) + '\t' + str(volume) + '\n')
    
    assert 1