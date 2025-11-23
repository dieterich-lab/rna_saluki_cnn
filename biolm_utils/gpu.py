"""GPU helper utilities extracted from params.py.

Expose get_detected_ngpus(cfg) to return the final computed GPU count for a
BioLMConfig.
"""

from .structured_config import BioLMConfig


def get_detected_ngpus(cfg: BioLMConfig) -> int:
    """Return the computed runtime GPU count from the config.

    This is a tiny helper that ensures callers don't need to inspect nested
    attributes themselves and always get a sane int >= 1.
    """
    value = getattr(getattr(cfg, "debugging", None), "detected_ngpus", None)
    try:
        ng = int(value) if value is not None else 1
        return ng if ng > 0 else 1
    except Exception:
        return 1
