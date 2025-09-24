import pytest
from ngsolve import *
from cosmos import *
import numpy as np
from cosmos.utils.generate_meshes import generate_boundary_cylinder

@pytest.fixture
def time_params():
    time_params = {}
    time_params['dt'] = 1e-3
    time_params['t_coef'] = Parameter(0)
    time_params['final_t'] = 1
    time_params['initial_t'] = 0
    return time_params

@pytest.fixture
def ch_params(ch_params_type):
    params = {}
    params['Neu_bnd_phase'] = 'bboundary'
    params['Neu_bnd_potential'] = 'bboundary'
    params['M'] = 0.01
    params['epsilon'] = 0.1
    params['sigma'] = 1
    if ch_params_type == 1:
        params['fes_order'] = 1
    if ch_params_type == 2:
        params['fes_order'] = 1
        params['bounds'] = [-0.999, 0.999]
    if ch_params_type == 3:
        params['fes_order'] = 1
        params['mass_preserving'] = True
    if ch_params_type == 4:
        params['fes_order'] = 1
        params['mass_preserving'] = True
        params['bounds'] = [-0.999, 0.999]
    if ch_params_type == 0:
        params['fes_order'] = 2
    return params

@pytest.mark.parametrize("ch_solver", [CahnHilliardBoundaryBachiniBDF1Model,
                                       CahnHilliardBoundaryBachiniLogBDF1Model])
@pytest.mark.parametrize("maxh", [0.1, 0.05])
@pytest.mark.parametrize("ch_params_type", [1, 2, 3, 4, 0])
def test_cahn_hilliard_cylinder_pol(
        request,
        cosmos_root,
        artifacts_path,
        time_params,
        ch_solver,
        ch_params, 
        maxh):
    
    out = artifacts_path
    filename = request.function.__name__

    def A_B(t):
        D = CF((1+0.3*sin(pi*t), 0, 0,\
                    0, 1+0.3*sin(pi*t), 0,\
                        0, 0, 1/(1+0.3*sin(pi*t))), dims = (3,3))
        # R = CF((cos(2*pi*t), -sin(2*pi*t), 0,\
        #             sin(2*pi*t), cos(2*pi*t), 0,\
        #                 0, 0, 1), dims = (3,3))
        # A = D*R
        A = D
        B = CF((0.2*t, 0.1*t, 0))
        return A, B
    A, B = A_B(time_params['t_coef'])
    detJ = Det(A)
    invA = Cof(A).trans/detJ
    inv_phi = invA*(CF((x,y,z)) - B) # Inverse of deformation map

    Ap, Bp = A_B(time_params['t_coef']+time_params['dt'])
    displacement_ex = (Ap-A)*inv_phi+(Bp-B)

    sample_rate = 10
    
    solvertime = SolverTime(**time_params)
    mesh = generate_boundary_cylinder(maxh = maxh)
    solvermesh = SolverMesh(mesh)
    solver = Solver(solvermesh, solvertime, iter=False,
                    name = filename, printing=True)
    solver.output_params(out, sample_rate=20)


    ch = ch_solver(solver, 1, input_params=ch_params)

    gfu = GridFunction(H1(mesh, definedon = mesh.Boundaries('.*')))
    size = len(gfu.vec.data)
    gfu.vec.data[:] = np.random.uniform(-0.999, 0.999, size)
    ch.phase.Set(gfu, definedon = mesh.Boundaries('.*'), dual = True)

    ale = ALEModel(solver=solver, model_order=2)
    ale.set_bnd_displacement(displacement_ex, 'default')
    
    solver.save_model_solution(ch)

    f_ch = open(os.path.join(out, 'ch_simulation.txt'), "w")
    f_ch.write('Time\tMass\tEnergy\tMaxValue\tMinValue\tArea\n')
    for _ in solver():
        if solver.time.iter%sample_rate == 0:
            gfu.Set(ch.phase, definedon = mesh.Boundaries('.*'), dual = True)
            ch_mass = Integrate(ch.phase, mesh, VOL_or_BND = BND)
            area = Integrate(1, mesh, VOL_or_BND = BND)
            ch_energy = ch.energy
            ch_minvalue = gfu.vec.FV().NumPy().min()
            ch_maxvalue = gfu.vec.FV().NumPy().max()
            f_ch.write(str(solver.time.t.Get()) + '\t' + str(ch_mass) + '\t' + str(ch_energy) 
                       + '\t' + str(ch_maxvalue) + '\t' + str(ch_minvalue) + '\t' + str(area) + '\n')