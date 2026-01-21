
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
from cosmos.pde.adr.boundary.adr_boundary_system_bdf1_model import ADRBoundarySystemBDF1Model

logging.getLogger().setLevel(logging.INFO)

@pytest.mark.parametrize("phi_cap", [0.969, 0.866, 0.5])
@pytest.mark.parametrize("D_factor", [1, 0.1])
@pytest.mark.parametrize("gamma_PMN_factor", [1, 10, 100])
def test_herant_3D_vp(
        request,
        artifacts_path,
        phi_cap, 
        D_factor,
        gamma_PMN_factor
    ):

    R_PMN = 4.25
    k_deg = 1
    k_prod = 10
    maxh_coarse = 2
    maxh_fine = 0.4
    F0 = 100
    gamma_drag = 100
    D_m = D_factor*10
    gamma_PMN = gamma_PMN_factor*1

    angle2_deg = 30
    angle2 = angle2_deg/180*pi

    pnt1 = occ.Pnt(0, 0, R_PMN)
    pnt2 = occ.Pnt(R_PMN, 0, 0)
    pnt3 = occ.Pnt(R_PMN*sin(angle2) , 0, -1*R_PMN*cos(angle2))
    pnt4 = occ.Pnt(R_PMN*sin(angle2/2) , 0, -1*R_PMN*cos(angle2/2))
    pnt5 = occ.Pnt(0 , 0, -R_PMN)

    arc1 = occ.ArcOfCircle(pnt1, pnt2, pnt3)
    arc2 = occ.ArcOfCircle(pnt3, pnt4, pnt5)
    seg1 = occ.Segment(pnt1, pnt5)

    w = occ.Wire([arc1, arc2, seg1])
    w.edges[1].name = 'pipette_bnd'
    w.edges[0].name = 'free_bnd'
    w.edges[0].maxh = maxh_fine
    f = occ.Face(w)
    body = f.Revolve(occ.Axis((0,0,0),occ.Z), 360).Rotate(occ.Axis((0,0,0),occ.Y), 90)
    body.edges[1].name = "bboundary"

    geo = occ.OCCGeometry(body)
    ngmesh = geo.GenerateMesh(maxh=maxh_coarse, uselocalh=True, optsteps2d=3)
    mesh = Mesh(ngmesh)

    ###################  Solver  ##################################
    t = Parameter(0)
    dt = Parameter(2e-3)
    root =  artifacts_path
    model_name = f"phi_cap{phi_cap}_d{D_m}_k{gamma_PMN}"
    model = CosmosModel(name=model_name, parentmesh=mesh, t0 = 0, t1 = 10,
                        dt = dt, t = t,
                        coupling_type = 'implicit', redistribute = True,
                        root = root, sample_rate = 50)

    ################### GRADIENT FLOW  ##################################
    comp1 = model.create_compartment(name = 'comp1', boundary = 'free_bnd', bboundary = 'bboundary', clamped_bbnd = "bboundary")
    geom_flow = model.create_pde(name = 'geom_flow', pde_model=GeometricalFlowModel, compartment=comp1)
    adr_bnd = model.create_pde(name = 'indicator', pde_model=ADRBoundarySystemBDF1Model, compartment=comp1, dim = 1)
    adr_bnd.set_params(
        Neu_bnd = 'default',
        b_1 = model.ale.wind,
        u0_1 = 1/(1+exp(-100*(x-R_PMN*phi_cap))),
    )

    ###################  VOLUME ADR  ##################################
    comp2 = model.create_compartment(name = 'comp2', material = 'default', boundary = 'free_bnd|pipette_bnd')
    adr_vol = model.create_pde(name = 'concentration', pde_model=ADRVolumeSystemBDF1Model, compartment=comp2, dim = 1)
    ns = specialcf.normal(3)
    adr_vol.set_params(
        Neu_bnd = 'free_bnd',
        d_1 = D_m,
        c_1 = k_deg,
        gradu_bnd_1 = lambda: IfPos(adr_bnd.sol[0], adr_bnd.sol[0], 0)*k_prod/D_m*ns,
        b_1 = model.ale.wind,
        printing = True
    )

    geom_flow.set_params(
        rhs = lambda: F0*adr_vol.gfu/gamma_drag,
        alpha = gamma_PMN/gamma_drag,
        volume_preserving = True
    )

    ###################  ALE  ##################################
    ale1 = model.create_ale('ale1', compartment=comp1, printing = False)
    ale1.set_normal_velocity(lambda: geom_flow.V_h)
    ale1.set_tangential_velocity(lambda: CF((0,0,0)))

    model.set_params(
        output_callables = {'energy': lambda: Integrate(0.5*(geom_flow.kappa_h - geom_flow.params['sp_curv'])**2, mesh, VOL_or_BND = BND),
                            'area': lambda: Integrate(1, mesh, VOL_or_BND = BND),
                            'volume': lambda: Integrate(1, mesh, VOL_or_BND = VOL),
                            'u_mass': lambda: Integrate(adr_vol.sol, mesh, VOL_or_BND = VOL)}
    )

    scene = Draw(adr_vol.gfu ,mesh)
    for _ in model():
        scene.Redraw()

    assert 1