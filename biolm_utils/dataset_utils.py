"""Small dataset helper utilities previously embedded inside cross_validation.py.

This module centralizes pure helpers for splitting datasets, creating subsets and
validating sizes so orchestration code (cross-validator / runner) stays concise
and easy to test.
"""

from collections import Counter
from typing import Iterable, List, Optional, Tuple

import numpy as np
from torch.utils.data import Subset


def split_indices(
    idx: Iterable[int], splitratio: List[int]
) -> Tuple[List[int], List[int], Optional[List[int]]]:
    """Split indices into train/val/(test) according to splitratio percentages.

    Args:
        idx: iterable of indices (already shuffled if randomness is desired)
        splitratio: list like [train_pct, val_pct] or [train_pct, val_pct, test_pct]

    Returns: tuple (train_idx, val_idx, test_idx|None)
    """
    idx = list(idx)
    n = len(idx)
    if len(splitratio) < 3:
        train_idx = idx[: int(n * splitratio[0] / 100)]
        val_idx = idx[-int(n * splitratio[1] / 100) :]
        test_idx = None
    else:
        train_end = int(n * splitratio[0] / 100)
        val_end = train_end + int(n * splitratio[1] / 100)
        train_idx = idx[:train_end]
        val_idx = idx[train_end:val_end]
        test_idx = idx[val_end:]
    return train_idx, val_idx, test_idx


def make_subsets(
    dataset,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: Optional[List[int]],
    dev: bool,
):
    """Create subsets for train/val/test. In dev/debug mode, all splits are set to full dataset.

    Returns (train_ds, val_ds, test_ds_or_none)
    """
    if not dev:
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx) if test_idx is not None else None
    else:
        idx = np.arange(len(dataset)).tolist()
        train_dataset = val_dataset = Subset(dataset, idx)
        test_dataset = Subset(dataset, idx) if test_idx is not None else None
    return train_dataset, val_dataset, test_dataset


def check_batchsize(ds, batchsize: int, name: str):
    """Raise if dataset is smaller than batch size (basic sanity check)."""
    if ds is not None and len(ds) < batchsize:
        raise ValueError(
            f"Size of the {name} dataset ({len(ds)}) is smaller than the batch size {batchsize}; lower the batch size."
        )


def log_classification_counts(
    params, dataset, train_dataset, val_dataset, test_dataset
):
    """Log class distributions for classification tasks.

    Kept as a pure helper so tests can call it easily.
    """
    if getattr(params, "task", None) == "classification":
        for name, ds in [
            ("train", train_dataset),
            ("val", val_dataset),
            ("test", test_dataset),
        ]:
            if ds is not None:
                counter = Counter(
                    [dataset.LE.classes_[dataset[x]["labels"]] for x in ds.indices]
                )
                # using print/logging is left to caller; return dict for tests
                print(f"{name} label distribution: {counter}")
        return True
    return False
