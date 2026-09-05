"""cosmos.core.utils.MandBP: bound- and mass-preserving vector corrections."""

import numpy as np
import pytest

from cosmos.core.utils import MandBP

pytestmark = pytest.mark.unit


def test_bound_preserving_mode_clips_to_the_given_interval():
    vec = np.array([-1.0, 0.5, 2.0])
    out = MandBP(vec, BP=[0.0, 1.0])
    assert np.array_equal(out, [0.0, 0.5, 1.0])


def test_mass_preserving_mode_conserves_the_weighted_sum():
    vec = np.array([0.2, 0.8, 1.1])
    weights = np.array([1.0, 1.0, 1.0])
    mass0 = float(np.sum(weights * vec))

    out = MandBP(vec, dt=0.1, weights=weights, BP=[0.0, 1.0], MP=True, mass0=mass0)

    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    assert np.sum(weights * out) == pytest.approx(mass0, abs=1e-8)
