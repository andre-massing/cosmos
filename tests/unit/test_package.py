"""Top-level package surface: ``cosmos.__init__``."""

import pytest

import cosmos

pytestmark = pytest.mark.unit


def test_version_is_non_empty_string():
    assert isinstance(cosmos.__version__, str)
    assert cosmos.__version__.strip()


def test_public_names_are_exported():
    for name in ("Config", "get_config", "set_config"):
        assert name in cosmos.__all__
        assert hasattr(cosmos, name)


def test_get_set_config_are_the_module_level_helpers():
    from cosmos.config.parameters import get_config, set_config

    assert cosmos.get_config is get_config
    assert cosmos.set_config is set_config
