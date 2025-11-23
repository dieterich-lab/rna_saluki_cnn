"""Compatibility validation helpers.

The project now keeps validation logic inside the structured dataclass
(`BioLMConfig.validate`). This module provides a small adapter so older
callers that previously imported a function from `params` can keep working by
importing from `biolm_utils.validation` if necessary.
"""

from .structured_config import BioLMConfig


def validate_config(cfg: BioLMConfig) -> None:
    """Validate a BioLMConfig instance (delegates to dataclass method)."""
    return cfg.validate()
