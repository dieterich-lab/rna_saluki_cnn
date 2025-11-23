"""Entry-point adapter for the Saluki plugin.

This module exposes the factory function referenced by the distribution
entry-point. The function simply re-exports the `get_saluki_config` factory
from the top-level `saluki` module in this repo.
"""

from __future__ import annotations

from saluki import get_saluki_config

__all__ = ["get_saluki_config"]
