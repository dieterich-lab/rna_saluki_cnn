"""Helpers to detect and migrate ambiguous `crossvalidation` config values.

This module provides utilities to analyze crossvalidation-related parameters and
optionally apply conservative automatic migrations when the config uses legacy
or ambiguous representations (e.g. boolean `true` used without splitpos).

The functions are intentionally conservative: they don't invent values unless
safe (e.g., convert `crossvalidation=0` -> False or `True` + `splitratio` ->
default k-fold), but they return human-readable recommendations and emit
warnings so users can inspect and commit explicit changes.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, List, Tuple

DEFAULT_K_FOLD = 5


def analyze_crossvalidation(params: Any) -> List[str]:
    """Return a list of suggested fixes / notes for the provided params.

    Accepts a structured `BioLMConfig` dataclass or objects that expose the
    same top-level attributes used by the cross-validation logic
    by the cross-validation logic (crossvalidation, splitpos, splitratio, devsplits).
    """
    notes: List[str] = []
    # Expect a structured config: attempt to read values from nested
    # data_source.* first (new style). Fall back to top-level attributes
    # only for permissive handling (should be removed eventually).
    ds = getattr(params, "data_source", None)
    if ds is not None:
        cv = getattr(ds, "crossvalidation", None)
        splitpos = getattr(ds, "splitpos", None)
        splitratio = getattr(ds, "splitratio", None)
        devsplits = getattr(ds, "devsplits", None)
    else:
        cv = getattr(params, "crossvalidation", None)
        splitpos = getattr(params, "splitpos", None)
        splitratio = getattr(params, "splitratio", None)
        devsplits = getattr(params, "devsplits", None)

    if cv is True:
        if not splitpos:
            # boolean true without splitpos is ambiguous
            notes.append(
                "crossvalidation=True without 'splitpos' is ambiguous. Consider either setting 'splitpos' + 'devsplits' for predefined splits, or choose an integer >=2 (k-fold) with 'splitratio'."
            )
        elif not devsplits:
            notes.append(
                "crossvalidation=True with 'splitpos' requires 'devsplits' (and optionally 'testsplits')."
            )

    if isinstance(cv, int):
        if cv < 2:
            notes.append(
                "Numeric crossvalidation < 2 is ambiguous/invalid. Use >= 2 or set to 0/False."
            )
        if splitpos is not None:
            notes.append(
                "Numeric crossvalidation conflicts with 'splitpos' (predefined splits). Remove one of them."
            )
        if not splitratio:
            notes.append(
                "Numeric crossvalidation should be paired with 'splitratio' (train/val/(test) percent list)."
            )

    if (not cv or cv == 0) and splitpos and not devsplits:
        notes.append(
            "'splitpos' is set but 'devsplits' is missing — provide devsplits or clear splitpos."
        )

    return notes


def migrate_crossvalidation(
    params: Any, auto_apply: bool = False, default_k: int = DEFAULT_K_FOLD
) -> Tuple[Any, List[str]]:
    """Return (new_params, notes). If auto_apply is True, apply conservative fixes.

    Safe auto-applications:
    - `crossvalidation == 0` -> set to False
    - `crossvalidation == True` and `splitratio` present -> set crossvalidation to default_k

    The function returns a (possibly updated) copy of params and notes describing
    actions or recommendations. Warnings are emitted for any fixes when applied.
    """
    notes: List[str] = []
    # Work on a deep copy of the provided structured config. We prefer the
    # nested data_source.* attributes when present.
    p = copy.deepcopy(params)
    ds = getattr(p, "data_source", None)
    if ds is not None:
        cv = getattr(ds, "crossvalidation", None)
        splitratio = getattr(ds, "splitratio", None)
    else:
        cv = getattr(p, "crossvalidation", None)
        splitratio = getattr(p, "splitratio", None)

    # Normalize zero to False
    if cv == 0:
        notes.append(
            "Converting numeric 0 to boolean False for crossvalidation (no CV)."
        )
        if auto_apply:
            if ds is not None:
                setattr(ds, "crossvalidation", False)
            else:
                setattr(p, "crossvalidation", False)
            warnings.warn("Auto-applied: crossvalidation 0 -> False")

    # If True but splitratio provided we can safely interpret as numeric k-fold
    if cv is True and splitratio:
        notes.append(
            f"Ambiguous True with splitratio set: recommending numeric k-fold crossvalidation={default_k}."
        )
        if auto_apply:
            if ds is not None:
                setattr(ds, "crossvalidation", default_k)
            else:
                setattr(p, "crossvalidation", default_k)
            warnings.warn(
                f"Auto-applied: crossvalidation True -> {default_k} since splitratio exists"
            )

    # No other changes are made automatically
    return p, notes


__all__ = ["analyze_crossvalidation", "migrate_crossvalidation", "DEFAULT_K_FOLD"]
