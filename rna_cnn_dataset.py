"""Minimal dataset implementation for the Saluki plugin.

This file intentionally stays small and focuses on the thin plugin contract —
it provides a Dataset class which can be referenced by the framework through
`Config.dataset_cls` without embedding framework logic into the plugin repo.
"""

from __future__ import annotations

from typing import Any, List

from torch.utils.data import Dataset


class SalukiDataset(Dataset):
    """Tiny dataset used for tests and demos.

    In production this is replaced with a real dataset class that handles
    loading sequences and preprocessing. The training and orchestration logic
    always lives in the framework (`biolm_utils`) and will call into this
    class as necessary.
    """

    def __init__(self, items: List[Any] | None = None):
        self._items = items or [0, 1, 2]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Any:
        return self._items[idx]
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder

from biolm_utils.rna_datasets import RNABaseDataset


class RNACNNDataset(RNABaseDataset):
    def __init__(self, **args):
        # ENFORCE SALUKI-INVARIANTS - fail fast before heavy superclass init
        # Saluki expects atomic one-hot encoding for the CNN input and a fixed
        # model sequence length. These are implementation invariants for Saluki
        # and must not be changed in global configs — fail fast if user overrides
        # them here so plugins stay portable and consistent.
        # args will be passed to RNABaseDataset; read from provided dict first
        # The constructor is intentionally permissive: callers may pass either
        # the structured `BioLMConfig`/compatible object or a plain dict of
        # keyword arguments. When called via `dataset_cls(tokenizer=..., args=args, ...)`
        # the `args` variable may be a dict with an `args` key holding the
        # config object — normalize that here so the invariant checks below are robust.
        if (
            isinstance(args, dict)
            and "args" in args
            and not isinstance(args.get("args"), dict)
        ):
            call_args = args.get("args")
        else:
            call_args = args

        encoding = (
            call_args.get("encoding")
            if isinstance(call_args, dict)
            else getattr(call_args, "encoding", None)
        )
        if encoding != "atomic":
            raise ValueError(
                "Saluki requires tokenization.encoding='atomic' (one-hot input). "
                "Do not set a different encoding in global configs; customize tokenization only in plugin-specific code."
            )

        expected_blocksize = 12288
        blocksize = (
            call_args.get("blocksize")
            if isinstance(call_args, dict)
            else getattr(call_args, "blocksize", None)
        )
        if blocksize != expected_blocksize:
            raise ValueError(
                f"Saluki requires training.blocksize={expected_blocksize}. "
                "This is an internal Saluki model property and cannot be changed via global configs."
            )

        # Only after invariants passed, initialize the base class
        super().__init__(**args)

        # Define a one-hot encoder
        non_special_vocab = [
            v
            for k, v in self.tokenizer.vocab.items()
            if k not in self.tokenizer.special_tokens_map.values()
        ]
        self.OHE = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.OHE.fit([[x] for x in non_special_vocab])

    def __getitem__(self, i):
        example = self.examples[i].copy()
        example["input_ids"] = self.OHE.transform(
            np.reshape(example["input_ids"], (-1, 1))
        )
        if self.args.specifiersep is not None:
            spec = self.specs[i]
            example["input_ids"] = np.concatenate((example["input_ids"], spec), axis=1)
        example["input_ids"] = torch.tensor(example["input_ids"], dtype=torch.float)
        return example
