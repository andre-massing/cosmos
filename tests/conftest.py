import os
import numpy as np
import pytest

@pytest.fixture(autouse=True)
def _set_random_seed():
    np.random.seed(12345)

@pytest.fixture(scope="session", autouse=True)
def _non_interactive_matplotlib():
    # Prevent GUI popups in visualization tests
    os.environ.setdefault("MPLBACKEND", "Agg")