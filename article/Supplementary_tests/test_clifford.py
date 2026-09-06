# %%

from ngsolve import *
from cosmos import *
from ngsolve.webgui import Draw
from cosmos.core.model import CosmosModel
from cosmos.pde.geom_flow.geometrical_flow_stationary_model import GeometricalFlowStationaryModel
from cosmos.pde.geom_flow.geometrical_flow_model import GeometricalFlowModel
import numpy as np
import logging
import pytest
import pandas as pd
from cosmos.utils.generate_meshes import generate_boundary_torus

logging.getLogger().setLevel(logging.INFO)


@pytest.mark.parametrize("h", [0.4, 0.2])
def test_clifford(request, artifacts_path, h):

    t = Parameter(0)
    root = artifacts_path
    model_name = f"test_clifford_h{h}"
    dt = 0.001

    mesh = generate_boundary_torus(maxh=h, R=2, r=1)
    model = CosmosModel(
        name=model_name,
        parentmesh=mesh,
        t0=0,
        dt=dt,
        t1=10,
        t=t,
        redistribute=True,
        root=root,
        samples=100,
        coupling_type="implicit",
        adaptive_timestep=True,
    )

    comp1 = model.create_compartment(name="comp1", boundary="default", bboundary="")
    geom_flow = model.create_pde(
        name="geom_flow", pde_model=GeometricalFlowModel, compartment=comp1, ale_type=0
    )
    ns = specialcf.normal(3)
    geom_flow.set_params(printing=True, volume_preserving=True)

    ale1 = model.create_ale("ale1", compartment=comp1)
    ale1.set_normal_velocity(lambda: geom_flow.V_h)
    ale1.set_tangential_velocity(lambda: CF((0, 0)))

    model.set_params(
        output_callables={
            "energy": lambda: Integrate(0.5 * (geom_flow.kappa_h) ** 2, mesh, VOL_or_BND=BND),
            "area": lambda: Integrate(1, mesh, VOL_or_BND=BND),
            "volume": lambda: Integrate(CF((x, 0, 0)) * specialcf.normal(3), mesh, VOL_or_BND=BND),
        }
    )

    model.run()

    assert 1


@pytest.mark.parametrize("h", [0.4, 0.2])
def test_clifford_stationary(request, artifacts_path, h):

    t = Parameter(0)
    root = artifacts_path
    model_name = f"test_clifford_stationary_h{h}"
    dt = 0.001

    mesh = generate_boundary_torus(maxh=h, R=2, r=1)
    model = CosmosModel(
        name=model_name,
        parentmesh=mesh,
        t0=0,
        dt=dt,
        t1=10,
        t=t,
        redistribute=True,
        root=root,
        samples=100,
        coupling_type="implicit",
        adaptive_timestep=True,
    )

    comp1 = model.create_compartment(name="comp1", boundary="default", bboundary="")
    geom_flow = model.create_pde(
        name="geom_flow", pde_model=GeometricalFlowStationaryModel, compartment=comp1, ale_type=0
    )
    ns = specialcf.normal(3)
    geom_flow.set_params(printing=True, volume_preserving=True, kappa0=CF(0))

    ale1 = model.create_ale("ale1", compartment=comp1)
    ale1.set_normal_velocity(lambda: geom_flow.V_h)
    ale1.set_tangential_velocity(lambda: CF((0, 0)))

    model.set_params(
        output_callables={
            "energy": lambda: Integrate(
                0.5 * (geom_flow.kappa_h - geom_flow.sp_curv_h) ** 2, mesh, VOL_or_BND=BND
            ),
            "area": lambda: Integrate(1, mesh, VOL_or_BND=BND),
            "volume": lambda: Integrate(CF((x, 0, 0)) * specialcf.normal(3), mesh, VOL_or_BND=BND),
        }
    )

    model.run()

    assert 1
