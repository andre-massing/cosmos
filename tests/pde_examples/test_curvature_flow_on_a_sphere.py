"""Example: curvature-driven flow of a sphere.

The first genuinely *coupled* example: the PDE
(:class:`~cosmos.pde.geom_flow.geometrical_flow_model.GeometricalFlowModel`)
computes a normal velocity from the surface's own curvature, and the ALE field
feeds that velocity straight back into the mesh, so the geometry the next step
is assembled on is the geometry the previous step produced. The wiring is::

    ale.set_normal_velocity(lambda: pde.V_h)      # PDE drives the mesh
    ale.set_tangential_velocity(lambda: CF(0,0,0))  # purely normal motion

Note the sign convention: with the outward normal these models produce the
*signed* mean curvature, so a unit sphere has ``kappa ~ -2/R``.

These are smoke tests with loose tolerances on a very coarse mesh, not a
convergence study -- see ``tests/pde/convergence`` for the latter.
"""

import numpy as np
import pytest
from ngsolve import CF

from cosmos.core.model import CosmosModel
from cosmos.pde.geom_flow.geometrical_flow_model import GeometricalFlowModel
from cosmos.utils.generate_meshes import generate_boundary_sphere

pytestmark = pytest.mark.example


def _build(t1, dt, maxh=0.5, **params):
    mesh = generate_boundary_sphere(maxh=maxh, R=1.0)
    model = CosmosModel("curvature_flow", mesh, t0=0.0, t1=t1, dt=dt, coupling_type="explicit")
    compartment = model.create_compartment("membrane", boundary="default", bboundary="")
    pde = model.create_pde("flow", GeometricalFlowModel, compartment, ale_type=0)
    if params:
        pde.set_params(**params)

    ale = model.create_ale("ale", compartment=compartment)
    ale.set_normal_velocity(lambda: pde.V_h)
    ale.set_tangential_velocity(lambda: CF((0.0, 0.0, 0.0)))
    return model, pde


def _vertex_radii(model):
    """Distance of every (ALE-moved) surface vertex from the origin."""
    coords = model.ale.X.vec.FV().NumPy()
    points = coords.reshape((len(coords) // 3, 3), order="F")
    return np.linalg.norm(points, axis=1)


def test_the_sphere_is_a_steady_state_of_willmore_flow():
    """alpha=1, beta=gamma=0: the round sphere is a critical point, so it stays put."""
    model, pde = _build(t1=0.05, dt=0.01)

    model.initialize()
    assert _vertex_radii(model) == pytest.approx(np.ones(model.parentmesh.nv))

    model.run()

    radii = _vertex_radii(model)
    kappa = pde.kappa_h.vec.FV().NumPy()
    assert np.isfinite(radii).all()
    assert np.isfinite(kappa).all()
    assert radii.mean() == pytest.approx(1.0, rel=0.05)
    assert radii.std() < 0.05  # it stays round
    assert kappa.mean() == pytest.approx(-2.0, rel=0.1)  # kappa = -2/R


def test_a_spontaneous_curvature_drives_the_sphere_to_its_preferred_radius():
    """With kappa_0 = -4 the energy is minimal at kappa = -4, i.e. R = 2/4 = 0.5."""
    model, pde = _build(t1=0.2, dt=0.02, sp_curv=CF(-4.0))

    model.run()

    radii = _vertex_radii(model)
    kappa = pde.kappa_h.vec.FV().NumPy()
    assert radii.mean() == pytest.approx(0.5, rel=0.15)
    assert radii.std() < 0.05
    assert kappa.mean() == pytest.approx(-4.0, rel=0.1)


def test_the_flow_shrinks_the_sphere_monotonically():
    model, pde = _build(t1=0.1, dt=0.02, sp_curv=CF(-4.0))

    history = []
    for _ in model():
        history.append(_vertex_radii(model).mean())

    assert len(history) > 2
    assert all(a > b for a, b in zip(history, history[1:]))
    assert history[0] == pytest.approx(1.0, rel=1e-6)
