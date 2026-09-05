"""cosmos.utils.mesh_fixing.CosmosAliasMesh: reading/editing Netgen .vol files."""

import pytest

from cosmos.utils.generate_meshes import generate_volume_circle
from cosmos.utils.mesh_fixing import CosmosAliasMesh

pytestmark = pytest.mark.unit


@pytest.fixture
def vol_file(tmp_path):
    mesh = generate_volume_circle(maxh=0.6)
    path = tmp_path / "disk.vol"
    mesh.ngmesh.Save(str(path))
    return path


def test_points_are_parsed_for_every_mesh_vertex(vol_file):
    alias = CosmosAliasMesh(str(vol_file))
    assert len(alias.get_points()) == alias.mesh.nv


def test_export_mesh_writes_a_non_empty_vol_file(vol_file, tmp_path):
    alias = CosmosAliasMesh(str(vol_file))
    out = tmp_path / "copy.vol"
    alias.export_mesh(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_an_unreadable_path_propagates_the_real_exception(tmp_path):
    with pytest.raises(Exception):
        CosmosAliasMesh(str(tmp_path / "does_not_exist.vol"))
