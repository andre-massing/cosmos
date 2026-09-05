"""cosmos.pde.distance.distance_volume_model.DistanceVolumeModel."""

import pytest

from cosmos.core.model import CosmosModel
from cosmos.pde import DistanceVolumeModel

pytestmark = pytest.mark.unit


def test_zero_bnd_is_required(disk_mesh):
    model = CosmosModel("m", disk_mesh, t0=0, t1=0.1, dt=0.1)
    comp = model.create_compartment("bulk", material="default", boundary="boundary")
    with pytest.raises(ValueError):
        model.create_pde("dist", DistanceVolumeModel, comp, ale_type=-1)


def test_constructs_and_is_volume_only(disk_mesh):
    model = CosmosModel("m", disk_mesh, t0=0, t1=0.1, dt=0.1)
    comp = model.create_compartment("bulk", material="default", boundary="boundary")
    dist = model.create_pde("dist", DistanceVolumeModel, comp, ale_type=-1, zero_bnd="boundary")
    assert dist.is_vol and not dist.is_bnd
