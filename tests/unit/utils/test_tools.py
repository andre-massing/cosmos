"""cosmos.utils.tools.gradient: projected gradient of a scalar field."""

import pytest
from ngsolve import CF, H1, GridFunction, Id, x, y

from cosmos.utils.generate_meshes import generate_volume_circle
from cosmos.utils.tools import gradient

pytestmark = pytest.mark.unit


def test_the_unprojected_gradient_of_x_squared_plus_y_squared_is_2x_2y():
    mesh = generate_volume_circle(maxh=0.6)
    f = x**2 + y**2
    grad_f = gradient(f, Id(2))

    gfu = GridFunction(H1(mesh, order=2) ** 2)
    gfu.Set(grad_f)

    point = mesh(0.3, 0.2)
    value = gfu(point)
    assert value[0] == pytest.approx(0.6, abs=1e-6)
    assert value[1] == pytest.approx(0.4, abs=1e-6)
