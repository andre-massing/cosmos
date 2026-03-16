from ngsolve import *
from cosmos import *
from ngsolve.webgui import Draw
from cosmos.utils.generate_meshes import generate_boundary_cigar, generate_boundary_sphere
from cosmos.core.model import CosmosModel
from cosmos.pde.willmore.geometrical_flow_stationary_model import GeometricalFlowStationaryModel
import numpy as np
import logging
import pytest
import pandas as pd

logging.getLogger().setLevel(logging.INFO)

@pytest.fixture()
def Tend(shape):
    if shape == '31':
        return 1
    elif shape == '51':
        return 0.3

@pytest.mark.parametrize("maxh", [0.25, 0.125])
@pytest.mark.parametrize("shape", ['31', '51'])
def test_kappa0_sigar(
        request,
        artifacts_path,
        Tend,
        shape,
        maxh):

    dt = 1e-3
    root =  artifacts_path
    model_name = f"test_kappa0_sigar{shape}_maxh{maxh}"
    
    if shape == '31':
        mesh = generate_boundary_cigar(maxh = maxh, r=1, h=3)
        model = CosmosModel(name = model_name, parentmesh=mesh, t0 = 0,
                            dt = dt, t1=Tend, redistribute = True,
                            root = root, samples = 100,
                            coupling_type = 'implicit', adaptive_timestep = True)
        
        comp1 = model.create_compartment(name = 'comp1', boundary = 'default', bboundary = '')
        geom_flow = model.create_pde(name = 'geom_flow', pde_model=GeometricalFlowStationaryModel, compartment=comp1, ale_type = 0)
        geom_flow.set_params(
            printing = True,
            kappa0 = CF(-2)
        )

        ale1 = model.create_ale('ale1', compartment=comp1)
        ale1.set_normal_velocity(lambda: geom_flow.V_h)
        ale1.set_tangential_velocity(lambda: CF((0,0)))

        model.set_params(
            output_callables = {'energy': lambda: Integrate(0.5*(geom_flow.kappa_h - geom_flow.sp_curv_h)**2, mesh, VOL_or_BND = BND),
                                'area': lambda: Integrate(1, mesh, VOL_or_BND = BND),
                                'volume': lambda: Integrate(CF((x, 0, 0))*specialcf.normal(3), mesh, VOL_or_BND = BND)}
        )
    elif shape == '51':
        mesh = generate_boundary_cigar(maxh = maxh, r=1, h=5)
        model = CosmosModel(name = model_name, parentmesh=mesh, t0 = 0,
                            dt = dt, t1=Tend, redistribute = True,
                            root = root, samples = 100,
                            coupling_type = 'implicit', adaptive_timestep = True)
        
        comp1 = model.create_compartment(name = 'comp1', boundary = 'default', bboundary = '')
        geom_flow = model.create_pde(name = 'geom_flow', pde_model=GeometricalFlowStationaryModel, compartment=comp1, ale_type = 0)
        geom_flow.set_params(
            printing = True,
            kappa0 = CF(-3)
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

@pytest.mark.parametrize("maxh", [0.1, 0.05])
def test_kappa0_sphere(
        request,
        artifacts_path,
        maxh):

    dt = 1e-3
    root =  artifacts_path
    model_name = f"test_kappa0_sphere_maxh{maxh}"
    
    mesh = generate_boundary_sphere(maxh = maxh, R=1)
    model = CosmosModel(name = model_name, parentmesh=mesh, t0 = 0,
                        dt = dt, t1=2, redistribute = True,
                        root = root, samples = 100,
                        coupling_type = 'implicit', adaptive_timestep = True)
    
    comp1 = model.create_compartment(name = 'comp1', boundary = 'default', bboundary = '')
    geom_flow = model.create_pde(name = 'geom_flow', pde_model=GeometricalFlowStationaryModel, compartment=comp1, ale_type = 0)
    geom_flow.set_params(
        printing = True,
        kappa0 = CF(3*sin(pi*x))
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