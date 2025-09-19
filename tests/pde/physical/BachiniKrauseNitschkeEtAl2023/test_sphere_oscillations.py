from ngsolve import *
from cosmos import *
from ngsolve.webgui import Draw
from cosmos.utils.generate_meshes import generate_boundary_sphere
import numpy as np
import logging
import pytest

logging.getLogger().setLevel(logging.INFO)

@pytest.fixture
def ch_default_params():
    params = {}
    params['M'] = 0.001
    params['epsilon'] = 0.05
    params['sigma'] = 1.5*sqrt(2)
    params['fes_order'] = 1
    params['bounds'] = [-1, 1]
    params['mass_preserving'] = True
    return params

@pytest.fixture
def solver_time_params():
    params = {}
    params['dt'] = 1e-4
    params['initial_t'] = 0
    params['final_t'] = 0.1
    return params

@pytest.mark.parametrize("maxh", [0.08, 0.04, 0.02])
@pytest.mark.parametrize("tang_vel_bool", [0, 1])
def test_sphere_oscillations(
        request,
        artifacts_path,
        maxh, 
        tang_vel_bool,
        solver_time_params,
        ch_default_params):
    
    out = artifacts_path
    filename = request.function.__name__
    mesh = generate_boundary_sphere(R=1, maxh = maxh)
    solvertime = SolverTime(**solver_time_params)
    solvermesh = SolverMesh(mesh)
    solver = Solver(solvermesh, solvertime, iter=False,
                    name = filename, printing=True)
    
    solver.output_params(out, sample_rate=50)

    willmore = WillmoreBoundaryInexBDF1Model(solver, 1, input_params = {"autoupdate": True})
    ale = ALEModel(solver, 2)
    ch = CahnHilliardBoundaryBachiniBDF1Model(solver, 3, input_params=ch_default_params)

    ns = specialcf.normal(3)
    Ps = Id(3) - OuterProduct(ns, ns)
    Qs = OuterProduct(ns, ns)
    V = VectorH1(mesh, definedon = mesh.Boundaries('.*'))
    pre_normal = GridFunction(V)
    normal = GridFunction(V)
    def compute_W():
        pre_normal.Set(Normalize(ns), dual = True, definedon = mesh.Boundaries(".*"))
        normal.Set(Normalize(pre_normal), dual = True, definedon = mesh.Boundaries(".*"))
        return -grad(normal).Trace()
    def rhs():
        var1 = ch.input_params["sigma"]*ch.input_params["epsilon"]/2*Norm(grad(ch.phase).Trace())**2*willmore.gfu_k
        var2 = ch.input_params["sigma"]/ch.input_params["epsilon"]*(0.25*(ch.phase**2-1)**2)*willmore.gfu_k
        Weingarten = compute_W()
        ns = specialcf.normal(mesh.dim)
        var3 = -1*ch.input_params["sigma"]*ch.input_params["epsilon"]*InnerProduct(grad(ch.phase).Trace(), Weingarten*grad(ch.phase).Trace())*ns
        return var1+var2+var3
    willmore.set_input_fields({
        "rhs": rhs,
        "elasticity_modulus": 0.1,
    })

    def displ():
        return Qs*willmore.displacement
    ale.set_bnd_displacement(displ, 'default', redistribute=True, redistribute_type='DuanLi')
    
    if tang_vel_bool:
        ch.set_input_fields({
            "b": ale.wind-Ps*grad(willmore.multiplier).Trace()
        })
    else:
        ch.set_input_fields({
            "b": ale.wind
        })
    ch.phase.Set(sin(pi*x)*sin(pi*y)*sin(pi*z), definedon = mesh.Boundaries('.*'), dual = True)

    solver.save_model_solution(ch.name, "phase")
    solver.save_model_solution(ch.name, "potential")
    solver.save_model_solution(willmore.name, "displacement")
    solver.save_model_solution(willmore.name, "mean_curvature")
    f_ch = open(os.path.join(out, filename, ch.name, 'ch_simulation.txt'), "w")
    f_wm = open(os.path.join(out, filename, willmore.name, 'wm_simulation.txt'), "w")
    sample_rate = 20
    f_ch.write('Time\tMass\tEnergy\tMaxValue\tMinValue\n')
    f_wm.write('Time\tEnergy\tArea\tVolume\tTotEnergy\n')
    for _ in solver():
        wm_area = Integrate(1, mesh, VOL_or_BND = BND)
        if solver.time.iter%sample_rate == 0:
            ch_mass = Integrate(ch.phase, mesh, VOL_or_BND = BND)
            ch_energy = ch.energy
            ch_minvalue = ch.phase.vec.FV().NumPy().min()
            ch_maxvalue = ch.phase.vec.FV().NumPy().max()
            f_ch.write(str(solver.time.t.Get()) + '\t' + str(ch_mass) + '\t' + str(ch_energy) 
                       + '\t' + str(ch_maxvalue) + '\t' + str(ch_minvalue) + '\n')
            wm_energy = willmore.energy
            wm_volume = Integrate(CF((x,0,0))*specialcf.normal(3), mesh, VOL_or_BND = BND)
            f_wm.write(str(solver.time.t.Get()) + '\t' + str(wm_energy) + '\t' + str(wm_area)+ '\t' 
                       + str(wm_volume) + '\t' + str(wm_energy+ch_energy)+ '\n')
    
    assert 1