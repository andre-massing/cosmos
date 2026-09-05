"""Example: distance to the boundary of a disk.

:class:`~cosmos.pde.distance.volume.distance_volume_model.DistanceVolumeModel`
is a stationary auxiliary solve rather than a time stepper: it computes a
regularised distance field that vanishes on ``zero_bnd`` and grows inwards,
plus the normalised gradient of that field (a unit direction pointing away from
the wall). It is registered with ``ale_type=-1`` so that it runs once, ahead of
the other PDEs, on every step.

Physics being checked: on the unit disk the exact distance-to-boundary is
``1 - r``, so the field must be zero on the rim, maximal at the centre and
monotone along a radius. The companion ``gfu2`` field is the *outward* unit
direction (it is seeded with ``specialcf.normal`` on the wall and follows
``grad`` of a wall-peaked potential inside), so it must have unit length and
point away from the centre.
"""

import numpy as np
import pytest

from cosmos.core.model import CosmosModel
from cosmos.pde.distance.distance_volume_model import DistanceVolumeModel
from cosmos.utils.generate_meshes import generate_volume_circle

pytestmark = pytest.mark.example


@pytest.fixture
def solved():
    mesh = generate_volume_circle(maxh=0.2)
    model = CosmosModel("distance_disk", mesh, t0=0.0, t1=0.0, dt=1.0)
    compartment = model.create_compartment("bulk", material="default", boundary="boundary")
    pde = model.create_pde(
        "dist", DistanceVolumeModel, compartment, ale_type=-1, zero_bnd="boundary"
    )
    model.initialize()
    pde.Solve()
    return mesh, pde


def test_the_distance_vanishes_on_the_wall_and_peaks_at_the_centre(solved):
    mesh, pde = solved
    values = pde.gfu.vec.FV().NumPy()

    assert np.isfinite(values).all()
    assert values.max() > 0.0

    # Zero on the rim: those are exactly the constrained (non-free) dofs.
    free = pde.fes.FreeDofs()
    on_the_wall = np.array([not free[i] for i in range(pde.fes.ndof)])
    assert on_the_wall.any()
    assert values[on_the_wall] == pytest.approx(np.zeros(on_the_wall.sum()), abs=1e-12)

    # ... and largest in the middle, where the exact distance to the rim is 1.
    centre = pde.gfu(mesh(0.0, 0.0))
    assert centre == pytest.approx(1.0, rel=0.2)
    assert centre > pde.gfu(mesh(0.5, 0.0)) > pde.gfu(mesh(0.9, 0.0)) > 0.0


def test_the_distance_decreases_monotonically_towards_the_wall(solved):
    mesh, pde = solved
    along_a_radius = [pde.gfu(mesh(r, 0.0)) for r in (0.0, 0.25, 0.5, 0.75, 0.95)]

    assert all(a > b for a, b in zip(along_a_radius, along_a_radius[1:]))


def test_the_velocity_field_is_a_unit_vector_pointing_at_the_wall(solved):
    mesh, pde = solved

    assert pde.vtk_names == ["dist_distance", "dist_velocity"]
    for point in ((0.7, 0.0), (0.0, 0.7), (-0.7, 0.0), (0.0, -0.7)):
        outward = np.asarray(point) / np.linalg.norm(point)
        direction = np.asarray(pde.gfu2(mesh(*point)))

        assert np.linalg.norm(direction) == pytest.approx(1.0, rel=0.1)
        assert np.dot(direction, outward) > 0.8
