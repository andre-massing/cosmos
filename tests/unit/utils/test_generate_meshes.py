"""cosmos.utils.generate_meshes: a representative sample of the mesh factories.

Meshes are used constantly throughout the rest of the suite (via
``tests/unit/conftest.py`` and ``tests/pde_examples``), which already exercises
most of these generators indirectly. This file just checks the basic contract
(dimension, element count, named boundaries) directly, at coarse resolution.
"""

import pytest

from cosmos.utils.generate_meshes import (
    generate_boundary_box,
    generate_boundary_sphere,
    generate_volume_circle,
)

pytestmark = pytest.mark.unit


def test_generate_volume_circle_is_a_2d_volume_mesh_with_a_named_boundary():
    mesh = generate_volume_circle(maxh=0.6, bnd_name="rim")
    assert mesh.dim == 2
    assert mesh.ne > 0
    assert "rim" in mesh.GetBoundaries()


def test_generate_boundary_sphere_is_a_surface_only_3d_mesh():
    mesh = generate_boundary_sphere(maxh=0.8, R=2.0)
    assert mesh.dim == 3
    assert mesh.ne == 0  # no volume elements: surface mesh only
    assert mesh.nface > 0


def test_generate_boundary_box_produces_six_faces_worth_of_elements():
    mesh = generate_boundary_box(maxh=0.6, a=1, b=1, c=1)
    assert mesh.dim == 3
    assert mesh.nface > 0
