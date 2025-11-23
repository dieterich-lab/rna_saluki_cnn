"""Thin compatibility wrapper around the refactored config modules.

The file used to contain most of the configuration logic. That logic was
extracted to smaller modules (`loader`, `gpu`, `validation`) to make the
project easier to maintain. This module keeps the public surface stable by
re-exporting the public helpers.
"""

from .gpu import get_detected_ngpus
from .loader import load_config, parse_args
from .validation import validate_config as _validate_config

__all__ = ["load_config", "parse_args", "get_detected_ngpus", "_validate_config"]
