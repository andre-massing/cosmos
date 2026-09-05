"""Shared fixtures for the Cosmos unit-test suite.

Every mesh here is deliberately *tiny*: unit tests must exercise API contracts,
not numerical accuracy, so a handful of elements is plenty and keeps the whole
suite in the low-seconds range.

Meshes are function-scoped on purpose. Building a :class:`CosmosModel` calls
``mesh.SetDeformation(...)`` on the parent mesh, so sharing one mesh object
between tests would leak ALE state across them.
"""

from __future__ import annotations

import pytest

from cosmos.utils.generate_meshes import (
    generate_boundary_1D_circle,
    generate_boundary_sphere,
    generate_volume_ball,
    generate_volume_circle,
)


@pytest.fixture
def disk_mesh():
    """Coarse 2D volume mesh of the unit disk (material 'default', boundary 'boundary')."""
    return generate_volume_circle(maxh=0.6)


@pytest.fixture
def ball_mesh():
    """Coarse 3D volume mesh of the unit ball (material 'default', boundary 'boundary')."""
    return generate_volume_ball(maxh=0.8)


@pytest.fixture
def sphere_mesh():
    """Coarse 3D surface-only mesh of the unit sphere (boundary 'default')."""
    return generate_boundary_sphere(maxh=0.8)


@pytest.fixture
def curve_mesh():
    """Coarse 2D surface-only (polyline) mesh of the unit circle (boundary 'default')."""
    return generate_boundary_1D_circle(N=12)
