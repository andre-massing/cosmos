"""cosmos.core.step_manager.CosmosStepManager: construction and validation.

The actual step-solving loop (explicit and implicit coupling, ALE dispatch) is
exercised end-to-end by the real PDE solves in ``tests/pde_examples`` — this
file only covers the parts that aren't naturally reached from there.
"""

import pytest

from cosmos.core.model import CosmosModel

pytestmark = pytest.mark.unit


def test_default_coupling_type_is_explicit(disk_mesh):
    model = CosmosModel("m", disk_mesh, t0=0, t1=0.1, dt=0.1)
    model.create_compartment("bulk", material="default", boundary="boundary")
    model.initialize()
    assert model.step.coupling_type == "explicit"


def test_an_invalid_coupling_type_raises_value_error(disk_mesh):
    model = CosmosModel("m", disk_mesh, t0=0, t1=0.1, dt=0.1, coupling_type="bogus")
    model.create_compartment("bulk", material="default", boundary="boundary")
    with pytest.raises(ValueError):
        model.initialize()
