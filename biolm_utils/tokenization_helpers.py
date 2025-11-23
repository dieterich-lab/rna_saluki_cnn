"""Tokenization helper utilities shared by tokenizer and dataset code.

This module centralizes parsing of the `atomicreplacements` option and
produces a dictionary or structured values so the tokenizer and dataset code
can behave consistently.
"""

from __future__ import annotations

from typing import Dict, Optional


def parse_atomic_replacements(value: Optional[str]) -> Optional[Dict[str, str]]:
    """Parse an `atomicreplacements` string into a dict.

    The value may be None, a dict-like string (e.g. '{"a": "A"}'), or a dict
    already. We intentionally avoid using eval here and use ast.literal_eval to
    be safer.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        return value

    try:
        import ast

        parsed = ast.literal_eval(value)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        # be permissive and return None if we cannot parse
        return None

    return None
