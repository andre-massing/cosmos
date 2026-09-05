"""cosmos.core.field.Field: the uniform coefficient/callable wrapper."""

import pytest
from ngsolve import CF, H1, CoefficientFunction, GridFunction

from cosmos.core.field import Field
from cosmos.utils.generate_meshes import generate_volume_circle

pytestmark = pytest.mark.unit


def test_a_plain_number_is_wrapped_in_a_coefficient_function():
    field = Field(1.5)
    assert isinstance(field(), CoefficientFunction)


def test_a_coefficient_function_is_stored_as_is():
    cf = CF(2.0)
    field = Field(cf)
    assert field() is cf
    assert field.cf is cf


def test_a_callable_is_evaluated_lazily():
    calls = []

    def make_cf():
        calls.append(1)
        return CF(3.0)

    field = Field(make_cf)
    assert field.is_callable
    assert not calls
    field()
    assert calls == [1]


def test_an_unsupported_type_raises_type_error():
    with pytest.raises(TypeError):
        Field(object())


def test_the_cf_setter_rejects_an_unsupported_type_with_a_clean_message():
    field = Field(1.0)
    with pytest.raises(TypeError):
        field.cf = object()


def test_the_cf_setter_replaces_the_stored_gridfunction():
    mesh = generate_volume_circle(maxh=0.6)
    gfu = GridFunction(H1(mesh, order=1))
    field = Field(1.0)
    field.cf = gfu
    assert field.cf is gfu
