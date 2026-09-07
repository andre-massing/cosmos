"""cosmos.core.model.CosmosModel: construction, kwargs validation, registration."""

import pytest
from ngsolve import CF

from cosmos.core.model import CosmosModel

pytestmark = pytest.mark.unit


def test_an_unknown_keyword_argument_is_rejected(disk_mesh):
    with pytest.raises(ValueError, match="surfac_ALE"):
        CosmosModel("m", disk_mesh, t0=0, t1=1, dt=0.1, surfac_ALE="mdr")


@pytest.mark.parametrize(
    "extra_kwarg",
    ["volume_ALE", "surface_ALE", "output_callables"],
)
def test_every_documented_keyword_is_accepted(disk_mesh, extra_kwarg):
    values = {"volume_ALE": "linel", "surface_ALE": "mdr", "output_callables": {}}
    CosmosModel("m", disk_mesh, t0=0, t1=1, dt=0.1, **{extra_kwarg: values[extra_kwarg]})


def test_mesh_derived_attributes(disk_mesh):
    model = CosmosModel("m", disk_mesh, t0=0, t1=1, dt=0.1)
    assert model.is_vol and not model.is_bnd
    assert "default" in model.vol_ids
    assert "boundary" in model.bnd_ids


def test_create_compartment_rejects_a_duplicate_name(disk_mesh):
    model = CosmosModel("m", disk_mesh, t0=0, t1=1, dt=0.1)
    model.create_compartment("bulk", material="default", boundary="boundary")
    with pytest.raises(ValueError):
        model.create_compartment("bulk", material="default", boundary="boundary")


def test_create_ale_rejects_a_second_ale_on_the_same_domain(sphere_mesh):
    model = CosmosModel("m", sphere_mesh, t0=0, t1=1, dt=0.1)
    comp = model.create_compartment("membrane", boundary="default", bboundary="")
    model.create_ale("ale1", comp)
    with pytest.raises(ValueError, match="already been set"):
        model.create_ale("ale2", comp)


def test_create_ale_rejects_a_second_ale_on_the_boundary_domain(disk_mesh):
    model = CosmosModel("m", disk_mesh, t0=0, t1=1, dt=0.1)
    comp = model.create_compartment("bulk", material="default", boundary="boundary")
    comp1 = model.create_compartment("surface", boundary="boundary", bboundary="default")
    model.create_ale("ale1", comp)
    with pytest.raises(ValueError, match="already been set"):
        model.create_ale("ale2", comp1)


def test_print_model_data_and_print_step_data_do_not_raise_bulk(disk_mesh):
    model = CosmosModel("m", disk_mesh, t0=0, t1=0.1, dt=0.1)
    model.create_compartment("bulk", material="default", boundary="boundary")
    model.print_model_data()
    model.initialize()
    model.io.save_step_data()
    model.print_step_data()


def test_print_model_data_and_print_step_data_do_not_raise_surface(sphere_mesh):
    model = CosmosModel("m", sphere_mesh, t0=0, t1=0.1, dt=0.1)
    model.create_compartment("surface", boundary="default", bboundary="")
    model.print_model_data()
    model.initialize()
    model.io.save_step_data()
    model.print_step_data()
