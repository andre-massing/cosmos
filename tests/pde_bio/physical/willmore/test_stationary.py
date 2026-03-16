from ngsolve import *
from cosmos import *
from ngsolve.webgui import Draw
from cosmos.core.model import CosmosModel
from cosmos.pde.willmore.geometrical_flow_stationary_model import GeometricalFlowStationaryModel
import numpy as np
import logging
import pytest
import pandas as pd

logging.getLogger().setLevel(logging.INFO)

@pytest.mark.parametrize("dt", [0.1, 0.01, 0.001])
def test_stationary_closed(
        request,
        artifacts_path,
        dt):

    root =  artifacts_path
    model_name = f"test_stationary_closed_dt{dt}"

    mesh = Mesh('../../../../data/bio/vol/spine_sliced/closed/spine_sliced_PM_closed_fixed.vol')
    model = CosmosModel(name = model_name, parentmesh=mesh, t0 = 0,
                        dt = dt, t1=1, redistribute = True,
                        root = root, samples = 100,
                        coupling_type = 'implicit', adaptive_timestep = True)
    
    comp1 = model.create_compartment(name = 'comp1', boundary = 'default', bboundary = '')
    geom_flow = model.create_pde(name = 'geom_flow', pde_model=GeometricalFlowStationaryModel, compartment=comp1, ale_type = 0)
    geom_flow.set_params(
        printing = True,
    )

    ale1 = model.create_ale('ale1', compartment=comp1)
    ale1.set_normal_velocity(lambda: geom_flow.V_h)
    ale1.set_tangential_velocity(lambda: CF((0,0)))

    model.set_params(
        output_callables = {'energy': lambda: Integrate(0.5*(geom_flow.kappa_h - geom_flow.sp_curv_h)**2, mesh, VOL_or_BND = BND),
                            'area': lambda: Integrate(1, mesh, VOL_or_BND = BND),
                            'volume': lambda: Integrate(CF((x, 0, 0))*specialcf.normal(3), mesh, VOL_or_BND = BND)}
        )
    
    model.run()

    assert 1

@pytest.mark.parametrize("dt", [0.1, 0.01, 0.001])
def test_stationary_open(
        request,
        artifacts_path,
        dt):

    root =  artifacts_path
    model_name = f"test_stationary_open_dt{dt}"

    mesh = Mesh( '../../../../data/bio/vol/spine_sliced/open/spine_sliced_PM_fixed.vol')
    model = CosmosModel(name = model_name, parentmesh=mesh, t0 = 0,
                        dt = dt, t1=1, redistribute = True,
                        root = root, samples = 100,
                        coupling_type = 'implicit', adaptive_timestep = True)
    
    comp1 = model.create_compartment(name = 'comp1', boundary = 'default', bboundary = 'bboundary1', clamped_bbnd = 'bboundary1')
    geom_flow = model.create_pde(name = 'geom_flow', pde_model=GeometricalFlowStationaryModel, compartment=comp1, ale_type = 0)
    geom_flow.set_params(
        printing = True,
    )

    ale1 = model.create_ale('ale1', compartment=comp1)
    ale1.set_normal_velocity(lambda: geom_flow.V_h)
    ale1.set_tangential_velocity(lambda: CF((0,0)))

    model.set_params(
        output_callables = {'energy': lambda: Integrate(0.5*(geom_flow.kappa_h - geom_flow.sp_curv_h)**2, mesh, VOL_or_BND = BND),
                            'area': lambda: Integrate(1, mesh, VOL_or_BND = BND),
                            'volume': lambda: Integrate(CF((x, 0, 0))*specialcf.normal(3), mesh, VOL_or_BND = BND)}
        )
    
    model.run()

    assert 1