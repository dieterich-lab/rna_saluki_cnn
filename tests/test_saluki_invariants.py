import unittest.mock
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from saluki_plugin.dataset import SALUKI_BLOCKSIZE, RNACNNDataset


class MockTokenizer:
    def __init__(self):
        self.model_max_length = None
        self.backend_tokenizer = MagicMock()
        self.backend_tokenizer.normalizer.normalize_str.return_value = "ACGT"
        self.backend_tokenizer.pre_tokenizer.pre_tokenize_str.return_value = [
            ("ACGT", (0, 4))
        ]
        self.vocab = {"A": 0, "C": 1, "G": 2, "T": 3, "[PAD]": 4}
        self.special_tokens_map = {"pad_token": "[PAD]"}

    def __call__(self, *args, **kwargs):
        return {"input_ids": [[1, 2, 3, 4]]}

    def convert_ids_to_tokens(self, ids):
        return ["A", "C", "G", "T"]


@pytest.fixture
def mock_args():
    # Create a minimal config structure similar to what Hydra provides
    conf = OmegaConf.create(
        {
            "data_source": {
                "filepath": "dummy_path.txt",
                "stripheader": False,
                "columnsep": "\t",
                "idpos": 1,
                "labelpos": 2,
                "seqpos": 3,
            },
            "tokenization": {"encoding": "atomic", "vocabsize": 4, "minfreq": 0},
            "training": {
                # Crucially, blocksize is MISSING or None here, as in the user's failing case
                "blocksize": None,
                "scaling": "log",
            },
            "settings": {"data_pre_processing": {}},
            "mode": "fine-tune",
            "task": "regression",
        }
    )
    return conf


def test_saluki_enforces_blocksize_invariant(mock_args):
    """
    Test that RNACNNDataset forces the tokenizer's model_max_length to 12288
    even if the configuration does not specify a blocksize.
    """
    tokenizer = MockTokenizer()

    # Mock file reading since RNABaseDataset tries to read the file
    with patch("builtins.open", unittest.mock.mock_open(read_data="id1\t1.0\tACGT\n")):
        # Instantiate the dataset
        dataset = RNACNNDataset(
            tokenizer=tokenizer, args=mock_args, add_special_tokens=False
        )

    # Assert that the invariant was enforced
    assert (
        tokenizer.model_max_length == SALUKI_BLOCKSIZE
    ), f"Saluki dataset did not enforce blocksize! Expected {SALUKI_BLOCKSIZE}, got {tokenizer.model_max_length}"


def test_saluki_rejects_non_atomic_encoding(mock_args):
    """Test that Saluki rejects non-atomic encoding."""
    mock_args.tokenization.encoding = "bpe"
    tokenizer = MockTokenizer()

    with pytest.raises(
        ValueError, match="Saluki requires tokenization.encoding='atomic'"
    ):
        with patch(
            "builtins.open", unittest.mock.mock_open(read_data="id1\t1.0\tACGT\n")
        ):
            RNACNNDataset(tokenizer=tokenizer, args=mock_args, add_special_tokens=False)
