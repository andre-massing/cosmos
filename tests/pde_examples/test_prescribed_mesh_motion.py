"""Example: prescribed ALE mesh motion, with no PDE at all.

The ALE machinery is usable on its own: give a surface compartment a normal
velocity and a tangential velocity, and the mesh moves. Each step is::

    model.ale.solve_ale()   # gather displacements, extend to the bulk
    model.time.next()       # advance t
    model.ale.finalize()    # commit Y -> mesh.deformation, Y -> Yo

Blowing a unit sphere outwards at unit speed must grow it by ``v*dt`` per step,
so after ``n`` steps its radius is ``1 + n*v*dt`` (up to the usual discrete-normal
error of a coarse triangulation).

The loop is driven by hand here because ``CosmosStepManager.solve_step`` returns
immediately when the model owns no PDEs, so plain ``model.run()`` would leave
the mesh where it started -- see the last test, which pins that behaviour down.
"""

import numpy as np
import pytest
from ngsolve import CF

from cosmos.core.model import CosmosModel
from cosmos.utils.generate_meshes import generate_boundary_sphere, generate_volume_circle

pytestmark = pytest.mark.example

DT = 0.01
SPEED = 1.0


def _build_sphere(normal_velocity=SPEED, tangential_velocity=(0.0, 0.0, 0.0)):
    mesh = generate_boundary_sphere(maxh=0.5, R=1.0)
    model = CosmosModel("ale_sphere", mesh, t0=0.0, t1=10 * DT, dt=DT)
    compartment = model.create_compartment("membrane", boundary="default", bboundary="")

    ale = model.create_ale("ale", compartment=compartment)
    ale.set_normal_velocity(CF(normal_velocity))
    ale.set_tangential_velocity(CF(tangential_velocity))
    return model, ale


def _vertex_radii(model):
    coords = model.ale.X.vec.FV().NumPy()
    points = coords.reshape((len(coords) // 3, 3), order="F")
    return np.linalg.norm(points, axis=1)


def _advance(model):
    model.ale.solve_ale()
    model.time.next()
    model.ale.finalize()


def test_a_constant_normal_velocity_inflates_the_sphere():
    model, _ = _build_sphere()
    model.initialize()
    assert _vertex_radii(model) == pytest.approx(np.ones(model.parentmesh.nv))

    for step in range(1, 4):
        _advance(model)
        radii = _vertex_radii(model)
        # The discrete normal of a coarse triangulation slightly under-resolves
        # the true one, hence the 5% tolerance on the growth.
        assert radii.mean() == pytest.approx(1.0 + step * SPEED * DT, rel=0.05)
        assert radii.std() < 1e-3  # the sphere stays round


def test_a_negative_normal_velocity_deflates_the_sphere():
    model, _ = _build_sphere(normal_velocity=-SPEED)
    model.initialize()

    for _ in range(3):
        _advance(model)

    assert _vertex_radii(model).mean() == pytest.approx(1.0 - 3 * SPEED * DT, rel=0.05)


def test_the_mesh_velocity_matches_the_displacement_over_the_time_step():
    model, _ = _build_sphere()
    model.initialize()
    _advance(model)

    displacement = model.ale.dY.vec.FV().NumPy()
    mesh_velocity = model.ale.W.vec.FV().NumPy()

    assert mesh_velocity == pytest.approx(displacement / DT)
    # Purely normal motion of unit speed: |W| ~ 1 everywhere.
    speeds = np.linalg.norm(mesh_velocity.reshape((len(mesh_velocity) // 3, 3), order="F"), axis=1)
    assert speeds.mean() == pytest.approx(SPEED, rel=0.05)


def test_a_velocity_must_be_prescribed_before_the_ale_can_solve():
    mesh = generate_boundary_sphere(maxh=0.8, R=1.0)
    model = CosmosModel("ale_incomplete", mesh, t0=0.0, t1=DT, dt=DT)
    compartment = model.create_compartment("membrane", boundary="default", bboundary="")
    ale = model.create_ale("ale", compartment=compartment)
    ale.set_normal_velocity(CF(SPEED))  # ... but no tangential velocity
    model.initialize()

    with pytest.raises(ValueError, match="Tangential velocity"):
        model.ale.solve_ale()


def test_a_volume_compartment_can_be_moved_by_a_domain_velocity():
    """The volume counterpart: a whole 2D disk translated along +x."""
    mesh = generate_volume_circle(maxh=0.5)
    model = CosmosModel("ale_disk", mesh, t0=0.0, t1=10 * DT, dt=DT)
    compartment = model.create_compartment("bulk", material="default", boundary="boundary")
    ale = model.create_ale("ale", compartment=compartment)
    ale.set_domain_velocity(CF((SPEED, 0.0)))
    model.initialize()

    coords = model.ale.X.vec.FV().NumPy()
    x_before = coords.reshape((len(coords) // 2, 2), order="F")[:, 0].copy()

    for _ in range(3):
        _advance(model)

    coords = model.ale.X.vec.FV().NumPy()
    x_after = coords.reshape((len(coords) // 2, 2), order="F")[:, 0]
    assert x_after - x_before == pytest.approx(np.full_like(x_before, 3 * SPEED * DT))


def test_running_the_model_without_a_pde_leaves_the_mesh_alone():
    """Documents a real limitation: the step manager skips ALE when no PDE exists.

    ``CosmosStepManager.solve_step`` is guarded by ``if len(model.pdes) > 0``, so
    a model whose only moving part is an ALE field never gets its mesh updated by
    ``model.run()``. Drive ``model.ale`` directly (as above) for that use case.
    """
    model, _ = _build_sphere()
    model.run()

    assert _vertex_radii(model) == pytest.approx(np.ones(model.parentmesh.nv))
