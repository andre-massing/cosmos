"""cosmos.core.io_manager: standalone VTK/PVD writer helpers.

``CosmosIOManager`` itself (the file-tree/VTKOutput side) is exercised
implicitly by every ``tests/pde_examples`` run that sets ``root=``/``samples=``;
here we only unit-test the small, pure writer functions.
"""

import numpy as np
import pytest

from cosmos.core.io_manager import write_curve_meshio, write_pvd

pytestmark = pytest.mark.unit


def test_write_curve_meshio_round_trips_point_scalars(tmp_path):
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    out = tmp_path / "curve.vtu"
    write_curve_meshio(str(out), points, point_scalars={"value": np.array([1.0, 2.0, 3.0])})
    assert out.exists() and out.stat().st_size > 0


def test_write_pvd_lists_every_file_with_its_timestep(tmp_path):
    out = tmp_path / "series.pvd"
    write_pvd(out, ["a.vtu", "b.vtu"], timesteps=[0.0, 0.5])
    text = out.read_text()
    assert "a.vtu" in text and "b.vtu" in text and "0.5" in text


def test_write_pvd_rejects_mismatched_timestep_count(tmp_path):
    with pytest.raises(ValueError):
        write_pvd(tmp_path / "series.pvd", ["a.vtu", "b.vtu"], timesteps=[0.0])
