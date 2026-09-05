# read version from installed package
from importlib.metadata import version

__version__ = version("cosmos")

from cosmos.config.parameters import Config, get_config, set_config

__all__ = ["Config", "get_config", "set_config"]
