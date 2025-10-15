import netgen.occ as occ
from ngsolve import *
from ngsolve.webgui import Draw
from cosmos import *
import logging
import pytest

logging.getLogger().setLevel(logging.INFO)

@pytest.fixture
def ch_default_params():
    params = {}
    params['M'] = 0.001
    params['epsilon'] = 0.1
    params['sigma'] = 1.5*sqrt(2)
    params['fes_order'] = 1
    params['bounds'] = [-0.999, 0.999]
    params['mass_preserving'] = True
    return params

@pytest.fixture
def solver_time_params():
    params = {}
    params['dt'] = 1e-4
    params['initial_t'] = 0
    params['final_t'] = 0.1
    return params

@pytest.mark.parametrize("kappa0", [0, -2, -4])
@pytest.mark.parametrize("pp", ["DuanLi", "MDR"])
def test_open_budding_sp_c(
        request,
        artifacts_path,
        solver_time_params,
        ch_default_params,
        pp,
        kappa0
    ):

    out = artifacts_path
    filename = request.function.__name__

    f1 = occ.WorkPlane(occ.Axes((0,0,0), n=occ.Z, h=occ.X)).Circle(0, 0, 1).Face()
    geo = occ.OCCGeometry(f1)
    mesh = Mesh(geo.GenerateMesh(maxh = 0.05))

    for point in mesh.ngmesh.Points():
        px, py = point[0], point[1]
        point[2] = 1+0.3*cos((sqrt(px**2+py**2))*pi)

    solvertime = SolverTime(**solver_time_params)
    solvermesh = SolverMesh(mesh)
    solver = Solver(solvermesh, solvertime, iter = False, printing=True,
                    name = filename)
    solver.output_params(out, sample_rate=10)

    conormal = Normalize(CF((x,y,0)))

    wm_default_params = {
        "clamped_bnd": 'default',
        "clamped_conormal": conormal,
        'autoupdate': True
    }
    willmore = WillmoreBoundaryInexBDF1Model(solver, 1, name = 'willmore_boundary_bdf1', input_params = wm_default_params)
    ale = ALEModel(solver, model_order=2)
    ch = CahnHilliardBoundaryBachiniLogBDF1Model(solver, 3, input_params=ch_default_params)

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
        "spontaneous_curvature": kappa0,
    })

    ch.set_input_fields({
        "b": ale.wind-Ps*grad(willmore.multiplier).Trace()
    })
    u0_ch = IfPos(sqrt(x**2+y**2)-0.5, -0.8, 0.8)
    ch.phase.Set(u0_ch, definedon = mesh.Boundaries('.*'), dual = True)

    def displ():
        return Qs*willmore.displacement
    ale.set_bnd_displacement(displ, 'default', redistribute=True, clamped_bnd='default', redistribute_type=pp)

    solver.save_model_solution(ch)
    solver.save_model_solution(willmore)
    f_ch = open(os.path.join(out, 'ch_simulation.txt'), "w")
    f_wm = open(os.path.join(out, 'wm_simulation.txt'), "w")
    sample_rate = 10
    f_ch.write('Time\tMass\tEnergy\tMaxValue\tMinValue\n')
    f_wm.write('Time\tEnergy\tArea\tTotEnergy\n')
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
            f_wm.write(str(solver.time.t.Get()) + '\t' + str(wm_energy) + '\t' + str(wm_area)+ '\t' 
                       +  '\t' + str(wm_energy+ch_energy)+ '\n')
    
    assert 1
