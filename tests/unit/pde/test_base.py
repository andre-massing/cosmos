"""cosmos.pde.base.BasePDEModel: the abstract PDE lifecycle contract."""

import pytest
from ngsolve import CF

from cosmos.core.field import Field
from cosmos.pde.base import BasePDEModel

pytestmark = pytest.mark.unit


class _DummyPDE(BasePDEModel):
    """Minimal concrete subclass: no-op lifecycle, one Field-typed param."""

    is_bnd = False
    is_vol = True

    def __init__(self, **kw):
        super().__init__(name="dummy", **kw)
        self.params["coef"] = Field(CF(0))

    def Initialize(self):
        pass

    def PreProcess(self):
        pass

    def Solve(self):
        pass

    def PostProcess(self):
        pass


def test_the_abstract_base_class_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BasePDEModel()


def test_set_params_updates_a_field_in_place_rather_than_replacing_it():
    pde = _DummyPDE()
    field_before = pde.params["coef"]
    pde.set_params(coef=CF(5))
    assert pde.params["coef"] is field_before
    assert field_before.cf.Compile()  # still a usable CoefficientFunction


def test_set_params_rejects_an_unknown_key():
    pde = _DummyPDE()
    with pytest.raises(ValueError):
        pde.set_params(not_a_real_param=1)
