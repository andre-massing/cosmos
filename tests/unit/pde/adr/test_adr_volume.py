"""cosmos.pde.adr.adr_volume_system_bdf1_model.ADRVolumeSystemBDF1Model."""

import pytest
from ngsolve import CF

from cosmos.core.model import CosmosModel
from cosmos.pde import ADRVolumeSystemBDF1Model

pytestmark = pytest.mark.unit


def test_dim_is_required(disk_mesh):
    model = CosmosModel("m", disk_mesh, t0=0, t1=0.1, dt=0.1)
    comp = model.create_compartment("bulk", material="default", boundary="boundary")
    with pytest.raises(ValueError):
        model.create_pde("adr", ADRVolumeSystemBDF1Model, comp, ale_type=1)


def test_dim_must_be_a_number(disk_mesh):
    model = CosmosModel("m", disk_mesh, t0=0, t1=0.1, dt=0.1)
    comp = model.create_compartment("bulk", material="default", boundary="boundary")
    with pytest.raises(TypeError):
        model.create_pde("adr", ADRVolumeSystemBDF1Model, comp, ale_type=1, dim="one")


def test_a_boundary_pde_cannot_be_placed_on_a_volume_compartment(disk_mesh):
    from cosmos.pde import ADRBoundarySystemBDF1Model

    model = CosmosModel("m", disk_mesh, t0=0, t1=0.1, dt=0.1)
    comp = model.create_compartment("bulk", material="default", boundary="boundary")
    with pytest.raises(ValueError):
        model.create_pde("adr", ADRBoundarySystemBDF1Model, comp, ale_type=1, dim=1)


def test_a_nonlinear_reaction_solves_for_a_scalar_system(disk_mesh):
    """Regression test: gfu_old has no .components for a scalar (dim=1) system."""
    model = CosmosModel("m", disk_mesh, t0=0, t1=0.02, dt=0.01)
    comp = model.create_compartment("bulk", material="default", boundary="boundary")
    adr = model.create_pde("adr", ADRVolumeSystemBDF1Model, comp, ale_type=1, dim=1)
    adr.set_params(d_1=CF(1e-2), u0_1=CF(1))
    adr.add_nonlinearity(target=1, expression="0.1 * u1 * v1")

    model.run()  # must not raise
