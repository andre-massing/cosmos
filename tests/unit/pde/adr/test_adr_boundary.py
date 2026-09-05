"""cosmos.pde.adr's two boundary ADR variants: stabilised and not."""

import pytest

from cosmos.core.model import CosmosModel
from cosmos.pde import ADRBoundarySystemBDF1Model, ADRBoundarySystemBDF1StabModel

pytestmark = pytest.mark.unit


def test_the_two_variants_are_genuinely_different_classes():
    """Regression test: both used to share the name ADRBoundarySystemBDF1Model."""
    assert ADRBoundarySystemBDF1Model is not ADRBoundarySystemBDF1StabModel
    assert ADRBoundarySystemBDF1Model.__name__ == "ADRBoundarySystemBDF1Model"
    assert ADRBoundarySystemBDF1StabModel.__name__ == "ADRBoundarySystemBDF1StabModel"


@pytest.mark.parametrize("cls", [ADRBoundarySystemBDF1Model, ADRBoundarySystemBDF1StabModel])
def test_dim_is_required(sphere_mesh, cls):
    model = CosmosModel("m", sphere_mesh, t0=0, t1=0.1, dt=0.1)
    comp = model.create_compartment("membrane", boundary="default", bboundary="")
    with pytest.raises(ValueError):
        model.create_pde("adr", cls, comp, ale_type=1)


@pytest.mark.parametrize("cls", [ADRBoundarySystemBDF1Model, ADRBoundarySystemBDF1StabModel])
def test_both_variants_are_surface_only(sphere_mesh, cls):
    model = CosmosModel("m", sphere_mesh, t0=0, t1=0.1, dt=0.1)
    comp = model.create_compartment("membrane", boundary="default", bboundary="")
    adr = model.create_pde("adr", cls, comp, ale_type=1, dim=1)
    assert adr.is_bnd and not adr.is_vol
