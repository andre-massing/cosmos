# %%

from ngsolve import *
from ngsolve.webgui import Draw
import netgen.occ as occ
from cosmos import *
import numpy as np
import logging
import pytest
from cosmos.core.model import CosmosModel
from cosmos.pde.willmore.geometrical_flow_model import GeometricalFlowModel
from cosmos.pde.adr.volume.adr_volume_system_bdf1_model import ADRVolumeSystemBDF1Model

logging.getLogger().setLevel(logging.INFO)

@pytest.mark.parametrize("phi_cap", [0.969, 0.866, 0.5])
@pytest.mark.parametrize("D_factor", [1, 0.1, 10])
@pytest.mark.parametrize("gamma_PMN_factor", [1, 0.1, 10])
def test_herant_2D_v1_ap(
        request,
        artifacts_path,
        phi_cap, 
        D_factor,
        gamma_PMN_factor
    ):

    R_PMN = 4.25
    k_deg = 1
    k_prod = 10
    maxh = 0.8
    F0 = 100
    gamma_drag = 100
    D_m = D_factor*10
    gamma_PMN = gamma_PMN_factor*1
    y_limit = sin(acos(phi_cap))*R_PMN
    trace = (1/(1+exp(-10*(y+y_limit))) - 1/(1+exp(-10*(y-y_limit))))*1/(1+exp(-2*x))

    angle2_deg = 30
    angle2 = angle2_deg/180*pi

    pnt1 = occ.Pnt(R_PMN, 0, 0)
    pnt2 = occ.Pnt(-R_PMN*cos(angle2), R_PMN*sin(angle2), 0)
    pnt3 = occ.Pnt(-R_PMN, 0, 0)
    pnt4 = occ.Pnt(-R_PMN*cos(angle2), -R_PMN*sin(angle2), 0)

    arc1 = occ.ArcOfCircle(pnt4, pnt1, pnt2)
    arc2 = occ.ArcOfCircle(pnt2, pnt3, pnt4)

    w = occ.Wire([arc1, arc2])
    f = occ.Face(w)
    f.maxh = maxh
    f.edges.maxh = 0.1
    f.edges[0].name = 'free_bnd'
    f.edges[1].name = 'pipette_bnd'

    geo = occ.OCCGeometry(f, dim = 2)

    ngmesh = geo.GenerateMesh(maxh=maxh,
        uselocalh=True,
        optsteps2d=3 )
    mesh = Mesh(ngmesh)

    ###################  Solver  ##################################
    t = Parameter(0)
    dt = Parameter(2e-3)
    root =  artifacts_path
    model_name = f"phi_cap{phi_cap}_d{D_m}_k{gamma_PMN}"
    model = CosmosModel(name=model_name, parentmesh=mesh, t0 = 0, t1 = 10,
                        dt = dt, t = t,
                        coupling_type = 'implicit', redistribute = True,
                        root = root, sample_rate = 25)

    ################### GRADIENT FLOW  ##################################
    comp1 = model.create_compartment(name = 'comp1', boundary = 'free_bnd', bboundary = 'default', clamped_bbnd = "default")
    pde1 = model.create_pde(name = 'geom_flow', pde_model=GeometricalFlowModel, compartment=comp1)

    ###################  VOLUME ADR  ##################################
    comp2 = model.create_compartment(name = 'comp2', material = 'default', boundary = 'free_bnd|pipette_bnd')
    pde2 = model.create_pde(name = 'concentration', pde_model=ADRVolumeSystemBDF1Model, compartment=comp2, dim = 1)
    ns = specialcf.normal(2)
    Draw(trace, mesh)
    flux = k_prod/D_m*ns*trace
    pde2.set_params(
        Neu_bnd = 'free_bnd',
        d_1 = D_m,
        c_1 = k_deg,
        gradu_bnd_1 = flux,
        b_1 = model.ale.wind,
    )

    pde1.set_params(
        rhs = lambda: F0*pde2.gfu/gamma_drag,
        alpha = gamma_PMN/gamma_drag,
        sp_curv = CF(-1/R_PMN),
        area_preserving = True
    )

    ###################  ALE  ##################################
    ale1 = model.create_ale('ale1', compartment=comp1)
    ale1.set_normal_velocity(lambda: pde1.V_h)
    ale1.set_tangential_velocity(lambda: CF((0,0)))

    model.set_params(
        output_callables = {'energy': lambda: Integrate(0.5*(pde1.kappa_h - pde1.params['sp_curv'])**2, mesh, VOL_or_BND = BND),
                            'area': lambda: Integrate(1, mesh, VOL_or_BND = BND),
                            'volume': lambda: Integrate(1, mesh, VOL_or_BND = VOL),
                            'u_mass': lambda: Integrate(pde2.sol, mesh, VOL_or_BND = VOL)}
    )

    scene = Draw(pde2.gfu ,mesh)
    for _ in model():
        scene.Redraw()

    assert 1