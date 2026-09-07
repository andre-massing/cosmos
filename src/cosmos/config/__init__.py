"""Re-exports the process-wide ``Config`` singleton accessors.

Also available directly from the top-level ``cosmos`` package
(``from cosmos import get_config``); both paths return the same object.
"""

from cosmos.config.parameters import Config, get_config, set_config

__all__ = ["Config", "get_config", "set_config"]
