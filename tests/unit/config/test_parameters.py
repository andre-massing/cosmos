"""cosmos.config.parameters: the shared Config singleton."""

import pytest

from cosmos.config.parameters import Config, get_config, set_config

pytestmark = pytest.mark.unit


def test_get_config_returns_the_shared_singleton():
    assert get_config() is get_config()
    assert isinstance(get_config(), Config)


def test_defaults():
    cfg = get_config()
    assert cfg.seed is None
    assert cfg.buffer == 2
    assert cfg.precision == "float64"


def test_set_config_updates_an_existing_attribute():
    original = get_config().precision
    try:
        set_config(precision="float32")
        assert get_config().precision == "float32"
    finally:
        set_config(precision=original)


def test_set_config_rejects_an_unknown_key():
    with pytest.raises(ValueError, match="precison"):
        set_config(precison="float32")
