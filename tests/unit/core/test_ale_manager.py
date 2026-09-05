"""cosmos.core.ale_manager.CosmosALEManager: construction and validation.

Actual mesh motion (normal/tangential velocities, displacement accumulation)
is exercised end-to-end in ``tests/pde_examples/test_prescribed_mesh_motion.py``.
"""

import pytest

from cosmos.core.model import CosmosModel

pytestmark = pytest.mark.unit


def test_default_volume_ale_is_laplace(disk_mesh):
    model = CosmosModel("m", disk_mesh, t0=0, t1=0.1, dt=0.1)
    assert model.ale.volume_ALE == "laplace"


def test_an_unknown_volume_ale_method_raises_value_error(disk_mesh):
    with pytest.raises(ValueError):
        CosmosModel("m", disk_mesh, t0=0, t1=0.1, dt=0.1, volume_ALE="not_a_real_method")


def test_a_surface_only_mesh_has_no_bulk_extension(sphere_mesh):
    model = CosmosModel("m", sphere_mesh, t0=0, t1=0.1, dt=0.1)
    assert not hasattr(model.ale, "A")
