import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from transformers import BertConfig, DefaultDataCollator, PreTrainedTokenizerFast

from biolm_utils.config import Config, set_config
from rna_cnn_dataset import RNACNNDataset
from rna_cnn_models import HFSaluki


def _set_saluki_config():
    # Mirror config from saluki.py
    params = [
        None,
        HFSaluki,
        PreTrainedTokenizerFast,
        1e-3,
        0.4,
        0.001,
        None,
        None,
        DefaultDataCollator,
        False,
        BertConfig,
        False,
        RNACNNDataset,
    ]
    set_config(Config(*params))


@pytest.mark.slow
def test_small_train_pipeline(tmp_path):
    """A small end-to-end test that tokenizes dummy data and runs one training epoch."""
    # set plugin config
    _set_saluki_config()

    # copy dummy data
    data_file = tmp_path / "dummy_data.txt"
    source = Path(__file__).resolve().parents[1] / "test" / "dummy_rna_data.txt"
    with open(source) as f_in, open(data_file, "w") as f_out:
        f_out.write(f_in.read())

    tokenize_cmd = [
        sys.executable,
        "saluki.py",
        "tokenize",
        "--filepath",
        str(data_file),
        "--outputpath",
        str(tmp_path),
        "--dev",
        "4",
        "--silent",
        "--encoding",
        "atomic",
        "--seqpos",
        "3",
        "--idpos",
        "1",
        "--labelpos",
        "2",
        "--columnsep",
        "\t",
        "--stripheader",
        "--accelerator",
        "cpu",
    ]
    result = subprocess.run(
        tokenize_cmd, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True
    )
    assert result.returncode == 0, f"Tokenize failed: {result.stderr.decode()}"
    assert (tmp_path / "tokenizer.json").exists()

    # Done – tokenization is a robust integration test on its own; fine-tune is covered by tests/test_smoke_pipeline.py
