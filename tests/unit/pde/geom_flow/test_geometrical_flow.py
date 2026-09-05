"""cosmos.pde.geom_flow: GeometricalFlowModel / GeometricalFlowStationaryModel.

A full Willmore-flow solve is exercised in
``tests/pde_examples/test_curvature_flow_on_a_sphere.py``; here we only cover
construction and defaults.
"""

import pytest

from cosmos.core.model import CosmosModel
from cosmos.pde import GeometricalFlowModel, GeometricalFlowStationaryModel

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("cls", [GeometricalFlowModel, GeometricalFlowStationaryModel])
def test_both_models_are_surface_only_with_default_coefficients(sphere_mesh, cls):
    model = CosmosModel("m", sphere_mesh, t0=0, t1=0.1, dt=0.1)
    comp = model.create_compartment("membrane", boundary="default", bboundary="")
    flow = model.create_pde("flow", cls, comp, ale_type=0)
    assert flow.is_bnd and not flow.is_vol
    assert flow.params["area_preserving"] is False
    assert flow.params["volume_preserving"] is False
