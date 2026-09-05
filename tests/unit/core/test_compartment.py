"""cosmos.core.compartment.CosmosCompartment: volume vs. surface sub-domains."""

import pytest

from cosmos.core.compartment import CosmosCompartment
from cosmos.core.model import CosmosModel

pytestmark = pytest.mark.unit


def test_a_volume_compartment_reports_the_model_dimension(disk_mesh):
    model = CosmosModel("m", disk_mesh, t0=0, t1=1, dt=0.1)
    comp = model.create_compartment("bulk", material="default", boundary="boundary")
    assert comp.is_vol and not comp.is_bnd
    assert comp.dim == model.dim


def test_a_surface_compartment_reports_one_dimension_lower(sphere_mesh):
    model = CosmosModel("m", sphere_mesh, t0=0, t1=1, dt=0.1)
    comp = model.create_compartment("membrane", boundary="default", bboundary="")
    assert comp.is_bnd and not comp.is_vol
    assert comp.dim == model.dim - 1


def test_an_unknown_material_name_raises_value_error(disk_mesh):
    model = CosmosModel("m", disk_mesh, t0=0, t1=1, dt=0.1)
    with pytest.raises(ValueError):
        model.create_compartment("bad", material="does_not_exist", boundary="boundary")


def test_neither_kwarg_pair_raises_value_error(disk_mesh):
    model = CosmosModel("m", disk_mesh, t0=0, t1=1, dt=0.1)
    with pytest.raises(ValueError):
        CosmosCompartment(name="bad", model=model)
