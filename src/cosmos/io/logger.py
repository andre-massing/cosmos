# Default logging configuration applied on package import (`from cosmos.io.logger
# import *` in cosmos/__init__.py); every module-level `logger` elsewhere in the
# package inherits this configuration unless overridden by the host application.

import logging

# Configure root logger (can also be done in the main script)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)