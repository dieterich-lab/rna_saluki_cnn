from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
from biolm.biolm_dataset import BioLMDataset
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset

# Saluki model constants
SALUKI_BLOCKSIZE = 12288
SALUKI_LABELPOS = 2


class RNACNNDataset(BioLMDataset):
    def __init__(self, **args):
        # enforce Saluki invariants early (one-hot encoding + fixed blocksize)
        if (
            isinstance(args, dict)
            and "args" in args
            and not isinstance(args.get("args"), dict)
        ):
            call_args = args.get("args")
        else:
            call_args = args

        self._ensure_saluki_labelpos(call_args)

        encoding = (
            call_args.tokenization.encoding
            if hasattr(call_args, "tokenization")
            else (
                call_args.get("encoding")
                if isinstance(call_args, dict)
                else getattr(call_args, "encoding", None)
            )
        )
        if encoding != "atomic":
            raise ValueError(
                "Saluki requires tokenization.encoding='atomic' (one-hot input). Do not set a different encoding in global configs; customize tokenization only in plugin-specific code."
            )

        expected_blocksize = SALUKI_BLOCKSIZE
        blocksize = (
            call_args.training.blocksize
            if hasattr(call_args, "training")
            else (
                call_args.get("blocksize")
                if isinstance(call_args, dict)
                else getattr(call_args, "blocksize", None)
            )
        )
        if blocksize is not None and blocksize != expected_blocksize:
            raise ValueError(
                f"Saluki requires training.blocksize={expected_blocksize}. This is an internal Saluki model property and cannot be changed via global configs."
            )

        # Ensure tokenizer uses the correct blocksize for padding
        tokenizer_instance = args.get("tokenizer")
        if tokenizer_instance is not None:
            tokenizer_instance.model_max_length = expected_blocksize
            if hasattr(tokenizer_instance, "init_kwargs"):
                tokenizer_instance.init_kwargs["model_max_length"] = expected_blocksize

        # Also inject it into the config args so any downstream logic sees it
        if "args" in args:
            config_args = args["args"]
            if hasattr(config_args, "training"):
                if config_args.training is None:
                    # If training config is missing, create a dummy one (unlikely but safe)
                    from biolm.structured_config import TrainingConfig

                    config_args.training = TrainingConfig()
                config_args.training.blocksize = SALUKI_BLOCKSIZE

        # initialize base class after invariants pass
        super().__init__(**args)

        non_special_vocab = [
            v
            for k, v in self.tokenizer.vocab.items()
            if k not in self.tokenizer.special_tokens_map.values()
        ]
        self.OHE = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.OHE.fit([[x] for x in non_special_vocab])

    def _ensure_saluki_labelpos(self, config_args: Any) -> None:
        data_source = getattr(config_args, "data_source", None)
        if data_source is None and isinstance(config_args, dict):
            data_source = config_args.setdefault("data_source", {})
        self._set_labelpos_if_missing(data_source)
        self._set_labelpos_if_missing(config_args)

    def _set_labelpos_if_missing(self, target: Any) -> None:
        if target is None:
            return
        if hasattr(target, "get") and hasattr(target, "__setitem__"):
            if target.get("labelpos") is None:
                target["labelpos"] = SALUKI_LABELPOS
            return
        if getattr(target, "labelpos", None) is None:
            setattr(target, "labelpos", SALUKI_LABELPOS)

    def __getitem__(self, i):
        example = self.examples[i].copy()
        example["input_ids"] = self.OHE.transform(
            np.reshape(example["input_ids"], (-1, 1))
        )
        specifiersep = (
            getattr(self.args.data_source, "specifiersep", None)
            if hasattr(self.args, "data_source")
            else getattr(self.args, "specifiersep", None)
        )
        if specifiersep is not None:
            spec = self.specs[i]
            example["input_ids"] = np.concatenate((example["input_ids"], spec), axis=1)
        example["input_ids"] = torch.tensor(example["input_ids"], dtype=torch.float)
        return example


# The canonical dataset implementation for this plugin is `RNACNNDataset`.
# The plugin entry-point exposes this class directly as the dataset implementation.
