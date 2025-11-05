# %%

from ngsolve import *
from ngsolve.webgui import Draw
import netgen.occ as occ
from cosmos import *
import numpy as np
import logging
import pytest

logging.getLogger().setLevel(logging.INFO)

@pytest.fixture
def solver_time_params():
    params = {}
    params['dt'] = 2e-3
    params['initial_t'] = 0
    params['final_t'] = 30
    return params

@pytest.mark.parametrize("angle_deg", [30, 45, 60, 90])
@pytest.mark.parametrize("kappa", [1, 0.01, 0.0001])
def test_phagocyosis_2D(
        request,
        artifacts_path,
        solver_time_params,
        kappa, 
        angle_deg,
    ):

    out = artifacts_path
    filename = request.function.__name__

    R = 1
    maxh = 0.2

    angle = angle_deg/180*pi

    pnt1 = occ.Pnt(R, 0, 0)
    pnt2 = occ.Pnt(R*cos(angle), R*sin(angle), 0)
    pnt3 = occ.Pnt(R*cos(angle), -R*sin(angle), 0)
    pnt4 = occ.Pnt(-R, 0, 0)

    arc1 = occ.ArcOfCircle(pnt3, pnt1, pnt2)
    arc2 = occ.ArcOfCircle(pnt2, pnt4, pnt3)

    w = occ.Wire([arc1, arc2])
    f = occ.Face(w)
    f.maxh = maxh
    f.edges.maxh = 0.05
    f.edges[0].name = 'bnd1'
    f.edges[1].name = 'bnd2'

    geo = occ.OCCGeometry(f, dim = 2)

    ngmesh = geo.GenerateMesh(maxh=maxh,
        uselocalh=True,
        optsteps2d=3 )
    mesh = Mesh(ngmesh)

    ###################  Solver  ##################################
    
    solvermesh = SolverMesh(mesh)
    solvertime = SolverTime(**solver_time_params)
    solver = Solver(solvermesh, solvertime, iter=True,
                    printing=True, name = filename)
    solver.output_params(out, sample_rate=150)

    ###################  SURFACE REACTIONS  ##################################
    ns = specialcf.normal(2)
    u = ADRVolumeBDF1Model(solver, 1, input_params={'Neu_bnd': 'bnd1'})

    ################### GRADIENT FLOW  ##################################
    willmore = WillmoreBoundaryInexBDF1Model(solver, 2)
    willmore.set_input_fields({
        'rhs': lambda: u.sol*ns,
        'elasticity_modulus': kappa

    })

    ###################  ALE  ##################################
    ale = ALEModel(solver, 3)
    ale.set_bnd_displacement(willmore.displacement, 'bnd1|bnd2', 
                            redistribute=True, 
                            redistribute_type='DuanLi')

    u.set_input_fields({
        'd': 1,
        'c': 1,
        'gradu_bnd': ns,
        'b': ale.wind
    })

    solver.save_model_solution(u)
    # solver.save_model_solution(ale)

    f_ch = open(os.path.join(out, 'u_simulation.txt'), "w")
    f_wm = open(os.path.join(out, 'wm_simulation.txt'), "w")
    sample_rate = 60
    f_ch.write('Time\tMass\tMaxValue\tMinValue\n')
    f_wm.write('Time\tEnergy\tArea\tVolume\n')
    for _ in solver():
        wm_area = Integrate(1, mesh, VOL_or_BND = BND)
        if solver.time.iter%sample_rate == 0:
            u_mass = Integrate(u.sol, mesh, VOL_or_BND = BND)
            u_minvalue = u.sol.vec.FV().NumPy().min()
            u_maxvalue = u.sol.vec.FV().NumPy().max()
            f_ch.write(str(solver.time.t.Get()) + '\t' + str(u_mass) + '\t' 
                       + str(u_maxvalue) + '\t' + str(u_minvalue) + '\n')
            wm_energy = willmore.energy
            wm_volume = Integrate(CF((x,0,0))*specialcf.normal(3), mesh, VOL_or_BND = BND)
            f_wm.write(str(solver.time.t.Get()) + '\t' + str(wm_energy) + '\t' + str(wm_area)+ '\t' 
                       + str(wm_volume) + '\n')
    
    assert 1

# %%
