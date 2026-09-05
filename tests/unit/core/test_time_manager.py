"""cosmos.core.time_manager: CosmosTimeManager / CosmosTimeHelper."""

import pytest
import numpy as np

from cosmos.core.time_manager import CosmosTimeHelper

pytestmark = pytest.mark.unit


def test_missing_required_keys_raises_value_error():
    with pytest.raises(ValueError):
        CosmosTimeHelper(t0=0, t1=1)  # dt missing


def test_a_non_numeric_t0_raises_type_error():
    with pytest.raises(TypeError):
        CosmosTimeHelper(t0="0", t1=1, dt=0.1)

def test_a_non_numeric_t1_raises_type_error():
    with pytest.raises(TypeError):
        CosmosTimeHelper(t0=0, t1="1", dt=0.1)

def test_a_non_positive_dt_raises_value_error():
    with pytest.raises(ValueError):
        CosmosTimeHelper(t0=0, t1=1, dt=-0.1)

@pytest.mark.parametrize(
    "timestep_list",
    [[0.1, 0.2, 0.3], np.array([0.1, 0.2, 0.3])],
)
def test_a_list_of_steps_is_accepted_and_advances_by_index(timestep_list):
    from ngsolve import Parameter

    helper = CosmosTimeHelper(t0=0, t1=1, dt=timestep_list)
    t, dt = Parameter(0), Parameter(0)
    helper.initialize(t, dt)
    assert dt.Get() == pytest.approx(0.1)
    helper.next(1, t, dt)
    assert dt.Get() == pytest.approx(0.2)
    helper.next(2, t, dt)
    assert dt.Get() == pytest.approx(0.3)
    helper.next(1, t, dt)
    with pytest.raises(AssertionError):
        assert dt.Get() == pytest.approx(0.3)
