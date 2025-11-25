"""Saluki plugin package.

This package contains the minimal plugin implementation used by the
integration test. It intentionally keeps the placeholder dataset/model
inside the package so the package is self-contained when installed.
"""

from .config import get_saluki_config

__all__ = ["get_saluki_config"]
