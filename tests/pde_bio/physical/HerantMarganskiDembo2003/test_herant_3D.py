
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
from cosmos.pde.adr.boundary.adr_boundary_system_bdf1_model_nostab import ADRBoundarySystemBDF1Model

logging.getLogger().setLevel(logging.INFO)

@pytest.mark.parametrize("gamma_tension", [100])
@pytest.mark.parametrize("gamma_curv", [0.1, 1, 10])
@pytest.mark.parametrize("phi_cap", [0.99, 0.96, 0.93])
def test_herant_3D_vp_clamped(
        request,
        artifacts_path,
        phi_cap, 
        gamma_curv,
        gamma_tension,
    ):

    R_PMN = 4.25
    k_deg = 1
    k_prod = 10
    F0 = 100
    D_m = 1
    gamma_drag = 1000

    angle2_deg = 20
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
    w.edges[0].maxh = 0.4
    f = occ.Face(w)
    body = f.Revolve(occ.Axis((0,0,0),occ.Z), 360).Rotate(occ.Axis((0,0,0),occ.Y), 90)
    body.edges[1].name = "bboundary"
    body.edges[1].maxh = 0.1

    geo = occ.OCCGeometry(body)
    ngmesh = geo.GenerateMesh(maxh=R_PMN, uselocalh=True, optsteps2d=3)
    mesh = Mesh(ngmesh)

    ###################  Solver  ##################################
    t = Parameter(0)
    dt = Parameter(0.02)
    root =  artifacts_path
    model_name = f"phi_cap{phi_cap}_alpha{gamma_curv}_gamma{gamma_tension}"
    model = CosmosModel(name=model_name, parentmesh=mesh, t0 = 0, t1 = 100,
                        dt = dt, t = t,
                        coupling_type = 'implicit', adaptive_timestep = False,
                        redistribute = True, surface_ALE = 'ms', volume_ALE = 'linel',
                        root = root, samples = 100)

    ################### GRADIENT FLOW  ##################################
    comp1 = model.create_compartment(name = 'comp1', boundary = 'free_bnd', bboundary = 'bboundary', clamped_bbnd = "bboundary")
    adr_bnd = model.create_pde(name = 'indicator', pde_model=ADRBoundarySystemBDF1Model, compartment=comp1, ale_type = 1, dim = 1)
    adr_bnd.set_params(
        Neu_bnd = 'bboundary',
        b_1 = lambda: model.ale.Vo,
        u0_1 = 1/(1+exp(-100*(x-R_PMN*phi_cap))),
        printing = True
    )

    ###################  VOLUME ADR  ##################################
    comp2 = model.create_compartment(name = 'comp2', material = 'default', boundary = 'free_bnd|pipette_bnd')
    adr_vol = model.create_pde(name = 'concentration', pde_model=ADRVolumeSystemBDF1Model, compartment=comp2, ale_type = 1, dim = 1)
    ns = specialcf.normal(3)
    adr_vol.set_params(
        Neu_bnd = 'free_bnd|pipette_bnd',
        d_1 = D_m,
        c_1 = k_deg,
        gradu_bnd_1 = lambda: IfPos(adr_bnd.sol[0], adr_bnd.sol[0], 0)*k_prod/D_m*ns*IfPos(t-50, 0, 1),
        b_1 = lambda: model.ale.Vo,
        printing = True
    )

    geom_flow = model.create_pde(name = 'geom_flow', pde_model=GeometricalFlowModel, compartment=comp1, ale_type = 0)
    geom_flow.set_params(
        kappa0 = CF(-2/R_PMN),
        sp_curv = CF(-2/R_PMN),
        rhs = lambda: F0*adr_vol.sol[0]/gamma_drag,
        alpha = gamma_curv/gamma_drag,
        gamma = gamma_tension/gamma_drag,
        volume_preserving = True,
        printing = True
    )

    ###################  ALE  ##################################
    ale1 = model.create_ale('ale1', compartment=comp1, printing = False)
    ale1.set_normal_velocity(lambda: geom_flow.V_h)
    ale1.set_tangential_velocity(lambda: CF((0,0,0)))

    model.set_params(
        output_callables = {'energy': lambda: Integrate(0.5*(geom_flow.kappa_h - geom_flow.params['sp_curv'])**2, mesh, VOL_or_BND = BND),
                            'area': lambda: Integrate(1, mesh, VOL_or_BND = BND),
                            'volume': lambda: Integrate(1, mesh, VOL_or_BND = VOL),
                            'u_mass': lambda: Integrate(adr_vol.sol[0], mesh, VOL_or_BND = VOL),
                            'i_mass': lambda: Integrate(adr_bnd.sol[0], mesh, VOL_or_BND = BND),
                            'x_bary': lambda: Integrate(x, mesh, VOL_or_BND = VOL),
                            'y_bary': lambda: Integrate(y, mesh, VOL_or_BND = VOL),
                            'z_bary': lambda: Integrate(z, mesh, VOL_or_BND = VOL),
                            'x_max': lambda: np.max(model.ale.X.components[0].vec.FV().NumPy()),
                            'y_max': lambda: np.max(model.ale.X.components[1].vec.FV().NumPy()),
                            'z_max': lambda: np.max(model.ale.X.components[2].vec.FV().NumPy())
                            }
    )

    model.run()

    assert 1