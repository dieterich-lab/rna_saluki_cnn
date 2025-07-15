import subprocess
import sys


def test_help_message_runs():
    """Check that the main script runs and prints help."""
    result = subprocess.run(
        [sys.executable, "-m", "saluki.py", "tokenize", "--help"], capture_output=True
    )
    assert result.returncode == 0
    assert b"usage" in result.stdout.lower()
