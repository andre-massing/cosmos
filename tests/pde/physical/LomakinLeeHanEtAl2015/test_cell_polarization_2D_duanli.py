# %%

from ngsolve import *
from ngsolve.webgui import Draw
from cosmos import *
import netgen.occ as occ
import numpy as np
import logging
import pytest

logging.getLogger().setLevel(logging.INFO)

@pytest.fixture
def solver_time_params():
    params = {}
    params['dt'] = 2e-2
    params['initial_t'] = 0
    params['final_t'] = 50
    return params

@pytest.fixture
def params():
    params = {}
    params['a_12'] = 1
    params['a_21'] = 1
    params['kappa1'] = 0.4
    params['kappa2'] = 0.4 
    params['m'] = 4
    params['s'] = 8
    return params

@pytest.mark.parametrize("angle_deg", [30, 45, 60, 90])
def test_cell_polarization_2D(
        request,
        artifacts_path,
        solver_time_params,
        params, 
        angle_deg,
    ):

    out = artifacts_path
    filename = request.function.__name__

    R = 1
    face = occ.WorkPlane(occ.Axes((0,0,0), n=occ.Z, h=occ.X)).Circle(0, 0, R).Face()
    face.edges.name = 'boundary'
    face.edges[0].maxh = 0.01
    geo = occ.OCCGeometry(face, dim = 2)
    mesh = geo.GenerateMesh(maxh=1, optsteps2d=3)
    mesh = Mesh(mesh)
    Area0 = Integrate(1, mesh, VOL_or_BND=BND)

    ###################  Solver  ##################################
    solvermesh = SolverMesh(mesh)
    solvertime = SolverTime(**solver_time_params)
    solver = Solver(solvermesh, solvertime, iter = True, 
                    printing=True, name = filename)
    solver.output_params(out, sample_rate=20)


    ###################  SURFACE REACTIONS  ##################################
    u = ADRBoundaryBDF1Model(solver, 1, input_params={'u0': 0.5}, name = 'u')
    w = ADRBoundaryBDF1Model(solver, 2, input_params={'u0': 0.5}, name = 'w')

    ###################  ALE  ##################################
    ale = ALEModel(solver, 3)
    def V():
        Area = Integrate(1, mesh, VOL_or_BND=BND)
        stretch = (Area**(params['s']+1))/(Area**params['s'] + Area0**params['s'])
        term1 = u.sol**params['m']/(u.sol**params['m'] + stretch**params['m'])
        term2 = w.sol**params['m']/(w.sol**params['m'] + 1)
        return term1 - term2
    ns = specialcf.normal(mesh.dim)
    def displ():
        return V()*solver_time_params['dt']*ns
    ale.set_bnd_displacement(displ, 'boundary', redistribute=True, redistribute_type='DuanLi')

    def u_rhs():
        term1 = params['kappa1']*V()*u.sol - u.sol**2 - params['a_12']*u.sol*w.sol
        return term1
    u.set_input_fields({
        'd': 0.1,
        'c': -1,
        'b': ale.wind,
        'rhs': u_rhs
    })
    
    def w_rhs():
        term1 = -params['kappa2']*V()*w.sol - w.sol**2 - params['a_21']*u.sol*w.sol
        return term1
    w.set_input_fields({
        'd': 0.1,
        'c': -1,
        'rhs': w_rhs,
        'b': ale.wind
    })

    solver.save_model_solution(ale)

    depletion = False
    f_u = open(os.path.join(out, 'u_simulation.txt'), "w")
    f_w = open(os.path.join(out, 'w_simulation.txt'), "w")
    f_u.write('Time\tMass\tMaxValue\tMinValue\n')
    f_w.write('Time\tMass\tMaxValue\tMinValue\n')
    sample_rate = 20
    for _ in solver():
        if solver.current_time>10 and depletion == False:
            gfu = GridFunction(w.gfu_old.space)
            gfu.vec.data = w.gfu.vec.data
            gfu.Set(IfPos(x/(sqrt(x**2+y**2))-cos(angle_deg/180*pi), 0.5*w.gfu, w.gfu), definedon = mesh.Boundaries('.*'))
            w.gfu_old.vec.data = gfu.vec.data
            w.gfu.vec.data = gfu.vec.data
            depletion = True
        if solver.time.iter%sample_rate == 0:
            u_mass = Integrate(u.sol, mesh, VOL_or_BND = BND)
            u_minvalue = u.sol.vec.FV().NumPy().min()
            u_maxvalue = u.sol.vec.FV().NumPy().max()
            f_u.write(str(solver.time.t.Get()) + '\t' + str(u_mass) + '\t' 
                       + str(u_maxvalue) + '\t' + str(u_minvalue) + '\n')
            w_mass = Integrate(w.sol, mesh, VOL_or_BND = BND)
            w_minvalue = w.sol.vec.FV().NumPy().min()
            w_maxvalue = w.sol.vec.FV().NumPy().max()
            f_w.write(str(solver.time.t.Get()) + '\t' + str(w_mass) + '\t' 
                       + str(w_maxvalue) + '\t' + str(w_minvalue) + '\n')
            
    assert 1
# %%
