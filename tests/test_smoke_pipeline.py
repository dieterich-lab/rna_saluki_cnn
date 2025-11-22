import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def test_help_message_runs():
    """Check that the main script runs and prints help."""
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    # make the bundled subtree importable when running the script from tests
    env["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        [sys.executable, "saluki.py", "tokenize", "--help"],
        capture_output=True,
        cwd=str(repo_root),
        env=env,
    )
    assert result.returncode == 0
    assert b"usage" in result.stdout.lower()


import os


@pytest.mark.skipif(
    os.environ.get("SKIP_HEAVY_TESTS", "1") == "1",
    reason="Skip full pipeline heavy tests by default",
)
def test_full_pipeline_integration():
    """Test the full pipeline: tokenize -> fine-tune with real data and models."""
    # Create a temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy dummy data
        data_file = temp_path / "test_data.txt"
        # Copy the bundled test data from the repo root (portable relative path)
        repo_root = Path(__file__).resolve().parents[1]
        bundled_data = repo_root / "test" / "dummy_rna_data.txt"
        shutil.copy(bundled_data, data_file)

        # 1. Tokenize with Hydra overrides
        tokenize_cmd = [
            sys.executable,
            "saluki.py",
            "tokenize",
            "--filepath",
            str(data_file),
            "--outputpath",
            str(temp_path),
            "--dev",
            "4",  # Use only 4 samples
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
        result = subprocess.run(tokenize_cmd, capture_output=True, cwd=str(repo_root))
        assert result.returncode == 0, f"Tokenize failed: {result.stderr.decode()}"

        # Check tokenizer was created
        assert (temp_path / "tokenizer.json").exists()

        # 2. Fine-tune
        finetune_cmd = [
            sys.executable,
            "saluki.py",
            "fine-tune",
            "--task",
            "regression",
            "--filepath",
            str(data_file),
            "--outputpath",
            str(temp_path),
            "--splitratio",
            "[50,25,25]",
            "--nepochs",
            "1",
            "--batchsize",
            "2",
            "--dev",
            "4",
            "--silent",
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
        result = subprocess.run(finetune_cmd, capture_output=True, cwd=str(repo_root))
        assert result.returncode == 0, f"Fine-tune failed: {result.stderr.decode()}"

        # Check predictions were generated during fine-tune
        # Predictions are saved per split (e.g., /fine-tune/0/test_predictions.csv)
        # Check predictions or model were created while the temp dir still exists
        predictions = list(temp_path.rglob("test_predictions.csv"))
        model_files = list(temp_path.rglob("pytorch_model.*"))
        assert (
            len(predictions) > 0 or len(model_files) > 0
        ), f"Neither predictions nor model files found under {temp_path}; check logs"
