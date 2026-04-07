# %%
from ngsolve import *
from ngsolve.webgui import Draw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cosmos import *

import pytest
import logging

logging.getLogger().setLevel(logging.INFO)

@pytest.mark.parametrize("Re", [7])
@pytest.mark.parametrize("surface_ALE", ['ms'])
@pytest.mark.parametrize("volume_ALE", ['laplace'])
@pytest.mark.parametrize("dt", [0.01, 0.001])
def test_spine3D_proto_3_init(
        request,
        artifacts_path,
        Re,
        surface_ALE,
        volume_ALE,
        dt
    ):

    print(request.path)
    from cosmos.utils.dendritic_spine_geom import generate_synapse3d
    mesh = generate_synapse3d(maxh = 0.1)

    ###################  PARAMETERS  ##################################

    A0 = 20
    B0 = 3000*3.6
    C0 = 40
    K_A = 0.0013
    K_B = 0.0081
    K_C = 0.0006
    I_A = 0.0255
    I_B = 24.4284
    I_C = 0.0237
    I_SA = 0.0293
    I_SB = 25.6684
    I_SC = 0.4384
    K_nuc = 0.0153
    K_sev = 0.012
    K_n = 0.6
    psi0 = 3.6
    psi1 = 0.02
    N = 3

    T = -59.75
    dt = Parameter(dt)
    t = Parameter(-60)

    from cosmos.core.model import CosmosModel

    root =  artifacts_path
    model_name = f"test_spine3D_proto_3_init_dt{dt.Get()}_volALE{volume_ALE}_surfALE{surface_ALE}_Re{Re}"
    model = CosmosModel(parentmesh=mesh, dt=dt, t=t, t0 = t.Get(), t1 = T,
                        root = root, samples = 400, name = model_name, coupling_type = 'implicit',
                        surface_ALE = surface_ALE, volume_ALE = volume_ALE,
                        redistribute = True, adaptive_timestep = True)
    
    model.print_model_data()

    ###################  BULK REACTIONS  ##################################
    from cosmos.pde.adr.volume.adr_volume_system_bdf1_model import ADRVolumeSystemBDF1Model
    from cosmos.pde.distance.volume.distance_volume_model import DistanceVolumeModel
    from cosmos.pde.willmore.geometrical_flow_stationary_model import GeometricalFlowStationaryModel

    comp1 = model.create_compartment('bulk', material = 'default', boundary = 'membrane|default')
    dist_fct = model.create_pde('distance_function', pde_model=DistanceVolumeModel, compartment=comp1, zero_bnd = 'membrane|default', ale_type = -1)
    dist_fct.set_params(printing = True)
    adr_sys = model.create_pde('adr_system', pde_model=ADRVolumeSystemBDF1Model, compartment=comp1, ale_type = 1, dim = 3)

    comp2 = model.create_compartment('surface', boundary = 'membrane', bboundary = 'membrane_bnd', clamped_bbnd = 'membrane_bnd')
    geom_flow = model.create_pde('willmore', pde_model=GeometricalFlowStationaryModel, compartment=comp2, ale_type = 0)
    geom_flow.set_params(
        rhs = lambda: adr_sys.sol[1]*1e-3,
        alpha = 1,
        printing = True
    )

    ale = model.create_ale('ale', compartment=comp2)
    ale.set_normal_velocity(geom_flow.V_h)
    ale.set_tangential_velocity(CF((0,0,0)))

    adr_sys.add_nonlinearity(target = 1, expression = 'K_nuc*psi1*u1*u2', map={'K_nuc': K_nuc, 'psi1': psi1})

    adr_sys.add_nonlinearity(target = 2, expression = '-1*psi0*(K_nuc*psi1*u1*u2 + K_sev*u3**N/(K_n + u3**N)*psi1*u2)', 
                            map={'K_nuc': K_nuc, 'psi1': psi1, 'psi0': psi0, 'K_sev': K_sev, 'K_n': K_n, 'N': N})

    adr_sys.add_nonlinearity(target = 3, expression = 'K_sev*u3**N/(K_n + u3**N)*psi1*u2',
                            map={'psi1': psi1, 'K_sev': K_sev, 'K_n': K_n, 'N': N})

    model.initialize()
    dist_fct.Initialize()
    dist_fct.Solve()
    id_funct = IfPos(dist_fct.sol[0]-0.02, 1, 0)*IfPos(z-0.4, 1, 0)
    impulse = IfPos(t, 1, 0)*IfPos(60-t, 1, 0)

    d = CF(1e-3)*exp(-Re)
    adr_sys.set_params(
        Neu_bnd = 'membrane|default',
        u0_1 = A0*id_funct,
        b_1 = lambda: model.ale.V,
        c_1 = K_A,
        d_1 = d,
        rhs_1 = I_A + I_SA*impulse,
        gradu_bnd_1 = CF((0,0,0)),
        bounds_1 = [0, 1e100],
        u0_2 = B0*id_funct,
        c_2 = K_B,
        b_2 = lambda: model.ale.V + dist_fct.sol[1]*1e-3*(1-exp(-Re)),
        d_2 = d,
        rhs_2 = psi0*(I_B + I_SB*impulse),
        gradu_bnd_2 = CF((0,0,0)),
        bounds_2 = [0, 1e100],
        u0_3 = C0*id_funct,
        b_3 = lambda: model.ale.V,
        c_3 = K_C,
        d_3 = d,
        rhs_3 = I_C + I_SC*impulse,
        gradu_bnd_3 = CF((0,0,0)),
        bounds_3 = [0, 1e100],
        printing = True
    )

    output_callables = {
        'mass_A': lambda: Integrate(adr_sys.sol[0], mesh),
        'mass_B': lambda: Integrate(adr_sys.sol[1], mesh),
        'mass_C': lambda: Integrate(adr_sys.sol[2], mesh),
        'energy': lambda: Integrate(0.5*(geom_flow.kappa_h - geom_flow.sp_curv_h)**2, mesh, VOL_or_BND = BND),
        'area': lambda: Integrate(1, mesh, VOL_or_BND = BND),
        'volume': lambda: Integrate(1, mesh, VOL_or_BND = VOL),
    }
    model.set_params(output_callables = output_callables)

    model.run()

    assert True