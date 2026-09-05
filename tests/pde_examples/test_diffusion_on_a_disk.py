"""Example: heat diffusion of a Gaussian bump on a disk.

The smallest complete Cosmos simulation there is — one volume compartment, one
scalar species, no advection, no reaction, no mesh motion. Everything goes
through the public :class:`~cosmos.core.model.CosmosModel` API:

    mesh -> model -> compartment -> pde -> model.run()

Physics being checked: a diffusion operator can only *smooth*. The peak must
come down, the trough must come up, the total mass is roughly conserved (no
flux is prescribed on the rim), and nothing blows up.
"""

import numpy as np
import pytest
from ngsolve import CF, exp, x, y

from cosmos.core.model import CosmosModel
from cosmos.pde.adr.adr_volume_system_bdf1_model import ADRVolumeSystemBDF1Model
from cosmos.utils.generate_meshes import generate_volume_circle

pytestmark = pytest.mark.example


@pytest.fixture
def diffusion():
    """A bump of heat on a coarse unit disk, diffused over five BDF1 steps."""
    mesh = generate_volume_circle(maxh=0.35)
    model = CosmosModel("diffusion_disk", mesh, t0=0.0, t1=0.05, dt=0.01, coupling_type="explicit")
    compartment = model.create_compartment("bulk", material="default", boundary="boundary")
    pde = model.create_pde("u", ADRVolumeSystemBDF1Model, compartment, ale_type=1, dim=1)
    pde.set_params(u0_1=exp(-10 * (x * x + y * y)), d_1=CF(0.5))
    return model, pde


def test_diffusion_smooths_the_bump_without_blowing_up(diffusion):
    model, pde = diffusion

    model.initialize()
    initial = pde.sol[0].vec.FV().NumPy().copy()
    assert initial.max() > 0.5  # the bump really is there

    model.run()
    final = pde.sol[0].vec.FV().NumPy()

    assert np.isfinite(final).all()
    assert final.max() < initial.max()  # the peak flattens
    assert final.min() > initial.min()  # the tails fill in
    assert final.min() >= 0.0  # no spurious undershoot


def test_diffusion_roughly_conserves_mass_with_no_flux_on_the_rim(diffusion):
    model, pde = diffusion

    model.initialize()
    weights = pde.weights0
    mass0 = float(np.sum(weights * pde.sol[0].vec.FV().NumPy()))

    model.run()
    mass1 = float(np.sum(weights * pde.sol[0].vec.FV().NumPy()))

    assert mass1 == pytest.approx(mass0, rel=0.05)


def test_a_uniform_field_is_a_steady_state_of_pure_diffusion():
    mesh = generate_volume_circle(maxh=0.35)
    model = CosmosModel("diffusion_flat", mesh, t0=0.0, t1=0.05, dt=0.01, coupling_type="explicit")
    compartment = model.create_compartment("bulk", material="default", boundary="boundary")
    pde = model.create_pde("u", ADRVolumeSystemBDF1Model, compartment, ale_type=1, dim=1)
    pde.set_params(u0_1=CF(2.0), d_1=CF(1.0))

    model.run()

    values = pde.sol[0].vec.FV().NumPy()
    assert values == pytest.approx(np.full_like(values, 2.0), abs=1e-8)
