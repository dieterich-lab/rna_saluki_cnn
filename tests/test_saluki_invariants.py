import types

import pytest

from rna_cnn_dataset import RNACNNDataset
from rna_cnn_models import HFSaluki


class DummyTokenizer:
    def __len__(self):
        return 10

    pad_token_id = 0


class DummyOHE:
    def __init__(self):
        # mimic sklearn OneHotEncoder.categories_ attribute
        self.categories_ = [range(4)]


class DummyDataset:
    def __init__(self):
        self.OHE = DummyOHE()
        self.nspecs = 0


def _make_args(blocksize=None, encoding=None):
    # simple namespace used by get_config and dataset init validation
    ns = types.SimpleNamespace()
    ns.blocksize = blocksize
    ns.encoding = encoding
    return ns


def test_hfsaluki_get_config_rejects_custom_blocksize():
    args = _make_args(blocksize=9999)
    tokenizer = DummyTokenizer()
    dataset = DummyDataset()

    with pytest.raises(ValueError):
        HFSaluki.get_config(
            args, config_cls=object, tokenizer=tokenizer, dataset=dataset, nlabels=1
        )


def test_hfsaluki_get_config_accepts_default_or_none():
    args = _make_args(blocksize=None)
    tokenizer = DummyTokenizer()
    dataset = DummyDataset()

    # Using transformers' config-like class (object here) will still be called - we simply test no exception
    # We'll use a lightweight callable that accepts kwargs and records them.
    class DummyCfg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    cfg = HFSaluki.get_config(
        args, config_cls=DummyCfg, tokenizer=tokenizer, dataset=dataset, nlabels=1
    )
    assert cfg.kwargs["max_position_embeddings"] == 12288


def test_rnacnn_dataset_rejects_non_atomic_encoding():
    bad_args = {"encoding": "bpe", "blocksize": 12288}
    with pytest.raises(ValueError):
        RNACNNDataset(**bad_args)


def test_rnacnn_dataset_rejects_custom_blocksize():
    bad_args = {"encoding": "atomic", "blocksize": 512}
    with pytest.raises(ValueError):
        RNACNNDataset(**bad_args)
