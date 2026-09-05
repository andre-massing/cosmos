from __future__ import annotations
import os
import numpy as np
import pytest
import re
from pathlib import Path


@pytest.fixture(autouse=True)
def _set_random_seed():
    np.random.seed(42)


@pytest.fixture(scope="session", autouse=True)
def _non_interactive_matplotlib():
    # Prevent GUI popups in visualization tests
    os.environ.setdefault("MPLBACKEND", "Agg")


def _project_root() -> Path:
    # Robust project-root discovery: walk up until we see a marker file/dir
    markers = {"pyproject.toml", "setup.cfg", ".git"}
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if any((parent / m).exists() for m in markers):
            return parent if parent.is_dir() else parent.parent
    return Path.cwd()


def _sanitize_nodeid(nodeid: str) -> str:
    # Make a safe folder name from something like "tests/pkg/test_big.py::TestX::test_y[param]"
    splitstring = nodeid.split("::", 1)[-1]
    s = re.sub(r"[/\\:]", "_", splitstring)  # path-ish separators
    s = re.sub(r"[^A-Za-z0-9_.+=@-]", "_", s)  # keep it reasonable
    return s


def pytest_addoption(parser):
    parser.addoption(
        "--artifacts-dir",
        action="store",
        default=None,
        help="Directory where test artifacts (large outputs) are stored.",
    )


@pytest.fixture(scope="session")
def artifacts_root(pytestconfig) -> Path:
    # Priority: CLI > ENV > default under repo root
    cli = pytestconfig.getoption("--artifacts-dir")
    env = os.getenv("ARTIFACTS_DIR")
    root = Path(cli or env or (_project_root() / "artifacts")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Optional: write a 'latest' pointer for convenience
    (root / "_README.txt").write_text(
        "Per-test artifacts live here. One subdirectory per test nodeid.\n", encoding="utf-8"
    )
    return root


@pytest.fixture(autouse=True)
def cosmos_root() -> Path:
    return Path(_project_root())


@pytest.fixture
def artifacts_path(request, artifacts_root: Path) -> Path:
    """
    A per-test directory you can write big outputs to, regardless of CWD.
    Usage in tests: artifacts_path / 'my_output.bin'
    """
    relpath = Path(request.fspath).relative_to(Path(_project_root() / "tests")).parent
    test_dir = artifacts_root / relpath / _sanitize_nodeid(request.node.nodeid)
    test_dir.mkdir(parents=True, exist_ok=True)
    # Optional: drop a manifest for quick provenance
    (test_dir / "_meta.txt").write_text(
        f"nodeid: {request.node.nodeid}\nfile: {request.fspath}\n", encoding="utf-8"
    )
    return test_dir
