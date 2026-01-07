"""
Full end-to-end Saluki pipeline test.

This test verifies the complete Saluki training pipeline:
1. Tokenization (creates tokenizer with atomic encoding)
2. Fine-tuning (trains CNN model on labeled regression task - 1 epoch)
3. Testing/prediction with Spearman correlation

Note: Saluki does not support pre-training (CNN architecture, not transformer).
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path

import pytest

logging.basicConfig(level=logging.DEBUG)


def debug_log(msg):
    """Print debug message to stdout for test visibility."""
    print(msg)


@pytest.fixture(scope="module")
def tiny_dataset():
    """Create minimal dataset for Saluki pipeline testing.

    Format: seq_id\tlabel\ta,t,g,c,... (comma-separated nucleotides)
    Saluki requires: columns 1=id, 2=label, 3=sequence
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="saluki_e2e_dataset_"))

    # 10 sequences, each exactly 100 nucleotides
    sequences_atgc = [
        (
            "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC",
            "1.5",
        ),
        (
            "GGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCUGGCU",
            "2.5",
        ),
        (
            "CCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGG",
            "3.5",
        ),
        (
            "UUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAA",
            "0.5",
        ),
        (
            "AAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUUAAUU",
            "4.5",
        ),
        (
            "GCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAGGCAG",
            "2.0",
        ),
        (
            "CUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAG",
            "3.0",
        ),
        (
            "GAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUGGAUG",
            "1.0",
        ),
        (
            "ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU",
            "1.2",
        ),
        (
            "UGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCA",
            "3.8",
        ),
    ]

    # Convert to comma-separated format for Saluki
    def to_comma_sep(seq):
        return ",".join(seq.lower())

    # Create training file - Saluki format: seq_id\tlabel\ta,t,g,c,...
    for split_name, indices in [("train", list(range(10)))]:
        filepath = tmpdir / f"{split_name}.txt"
        with open(filepath, "w") as f:
            for idx in indices:
                seq, label = sequences_atgc[idx]
                comma_sep = to_comma_sep(seq)
                f.write(f"seq_{idx}\t{label}\t{comma_sep}\n")

    debug_log(f"Created tiny dataset at {tmpdir}")
    return tmpdir


def run_command(cmd, cwd="/prj/RNA_NLP/biolm_utils", timeout=600):
    """Helper to run command and return result."""
    debug_log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout
    )
    if result.returncode != 0:
        debug_log(f"STDOUT:\n{result.stdout}")
        debug_log(f"STDERR:\n{result.stderr}")
    return result


def test_saluki_full_pipeline(tiny_dataset):
    """Test full Saluki pipeline: tokenize -> fine-tune -> test (no pre-training)."""
    debug_log("=" * 80)
    debug_log("STARTING SALUKI FULL PIPELINE TEST")
    debug_log("=" * 80)

    with tempfile.TemporaryDirectory(prefix="saluki_e2e_") as tmpdir:
        output_dir = Path(tmpdir)

        # Step 1: Tokenization (atomic encoding for Saluki)
        debug_log("\n>>> STEP 1: TOKENIZATION")
        tokenize_cmd = [
            "poetry",
            "run",
            "python",
            "-m",
            "biolm.runner",
            f"data_source.filepath={tiny_dataset}/train.txt",
            f"outputpath={output_dir}",
            "mode=tokenize",
            "model=saluki",
            "tokenization.vocabsize=100",
            "training.num_epochs=1",
            "debugging.accelerator=cpu",
        ]
        result = run_command(tokenize_cmd)
        assert result.returncode == 0, f"Tokenization failed:\n{result.stderr}"
        debug_log("✓ Tokenization completed")

        # Verify tokenizer was created
        tokenizer_file = output_dir / "tokenize" / "tokenizer.json"
        assert tokenizer_file.exists(), f"Tokenizer not found at {tokenizer_file}"
        debug_log(f"✓ Tokenizer created at {tokenizer_file}")

        # Step 2: Fine-tuning (Saluki goes directly to fine-tuning, no pre-training)
        debug_log("\n>>> STEP 2: FINE-TUNING (Saluki - no pre-training)")
        finetune_cmd = [
            "poetry",
            "run",
            "python",
            "-m",
            "biolm.runner",
            f"data_source.filepath={tiny_dataset}/train.txt",
            f"outputpath={output_dir}",
            "mode=fine-tune",
            "model=saluki",
            "task=regression",
            "training.num_epochs=1",
            "+training.blocksize=12288",  # Saluki requirement
            "model.num_layers=1",
            "model.kernel_size=3",
            "model.num_filters=32",
            "debugging.accelerator=cpu",
            "training.batchsize=2",
            "data_source.splitratio=[80,20]",
        ]
        result = run_command(finetune_cmd, timeout=900)
        assert result.returncode == 0, f"Fine-tuning failed:\n{result.stderr}"
        debug_log("✓ Fine-tuning completed")

        # Verify fine-tuning checkpoint
        finetune_dir = output_dir / "fine-tune"
        assert finetune_dir.exists(), f"Fine-tune directory not found"
        debug_log(f"✓ Fine-tuning checkpoint created")

        # Step 3: Testing
        debug_log("\n>>> STEP 3: TESTING")
        test_results_file = finetune_dir / "eval_results.json"
        assert test_results_file.exists(), f"Results file not found"

        with open(test_results_file) as f:
            results = json.load(f)

        debug_log(f"Results: {results}")
        assert "eval_spearman rho" in results, "Spearman correlation not in results"
        debug_log(f"✓ Test Spearman: {results['eval_spearman rho']}")

        debug_log("\n" + "=" * 80)
        debug_log("SALUKI FULL PIPELINE TEST PASSED")
        debug_log("=" * 80)
