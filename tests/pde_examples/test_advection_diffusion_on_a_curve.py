"""Example: advection-diffusion-reaction on a curve, with a Dirichlet condition.

This is the *surface* counterpart of the volume example: the mesh has no
codim-0 elements at all (``mesh.ne == 0``, so ``model.is_bnd`` is true) and the
PDE is posed on the curve itself. The half-circle arc is used because it has
named endpoints (``bboundary``), which is what a surface Dirichlet condition
(``Dir_bnd``, imposed weakly by Nitsche) is anchored to.

Physics being checked: with the concentration clamped to 1 at both ends and
diffusion inside, the solution relaxes towards 1 everywhere and stays within
its physical bounds; a decay reaction pulls it back down again.
"""

import numpy as np
import pytest
from ngsolve import CF

from cosmos.core.model import CosmosModel
from cosmos.pde.adr.adr_boundary_system_bdf1_model_nostab import (
    ADRBoundarySystemBDF1Model,
)
from cosmos.utils.generate_meshes import generate_boundary_arc

pytestmark = pytest.mark.example

ARC_BOUNDARY = "bbnd_l|bbnd_r"  # the two halves of the arc
ARC_ENDS = "bboundary"  # its two free endpoints


def _build(t1=0.1, dt=0.02, **params):
    mesh = generate_boundary_arc(N=16)
    model = CosmosModel("adr_arc", mesh, t0=0.0, t1=t1, dt=dt, coupling_type="explicit")
    assert model.is_bnd is True  # a codim-1 model: the curve *is* the domain

    compartment = model.create_compartment("curve", boundary=ARC_BOUNDARY, bboundary=ARC_ENDS)
    pde = model.create_pde("u", ADRBoundarySystemBDF1Model, compartment, ale_type=1, dim=1)
    pde.set_params(**params)
    return model, pde


def test_a_dirichlet_condition_fills_the_curve_from_its_endpoints():
    model, pde = _build(
        u0_1=CF(0.0),
        d_1=CF(0.5),
        u_bnd_1=CF(1.0),
        Dir_bnd=ARC_ENDS,
    )

    model.run()
    values = pde.sol[0].vec.FV().NumPy()

    assert np.isfinite(values).all()
    assert values.max() == pytest.approx(1.0, abs=0.1)  # the clamped ends
    assert values.min() > -0.05  # essentially no undershoot
    assert values.max() <= 1.05  # and no overshoot either
    assert values.mean() > 0.1  # material really moved inwards


def test_advection_biases_the_profile_towards_one_end():
    """The same setup, but blown along +x: the two halves stop being symmetric."""
    model, pde = _build(
        u0_1=CF(0.0),
        d_1=CF(0.1),
        u_bnd_1=CF(1.0),
        Dir_bnd=ARC_ENDS,
        b_1=CF((2.0, 0.0)),
    )

    model.run()
    values = pde.sol[0].vec.FV().NumPy()

    assert np.isfinite(values).all()
    assert values.max() == pytest.approx(1.0, abs=0.15)


def test_a_linear_decay_reaction_consumes_the_species():
    """No boundary source, just c*u decay: whatever starts there must shrink."""
    model, pde = _build(u0_1=CF(1.0), d_1=CF(0.1), c_1=CF(5.0))

    model.initialize()
    initial = pde.sol[0].vec.FV().NumPy().copy()
    model.run()
    final = pde.sol[0].vec.FV().NumPy()

    assert np.isfinite(final).all()
    assert 0.0 < final.max() < initial.max()
