import os
import subprocess
import sys


def test_quick_train_script_runs_ok(tmp_path):
    # Run the example script as a subprocess using the current test Python
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script = os.path.join(repo_root, "examples", "quick_train_saluki.py")

    assert os.path.exists(script), f"Example script not found: {script}"

    proc = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=repo_root,
    )

    if proc.returncode != 0:
        # Show the output to help debugging in CI
        print("--- STDOUT ---\n", proc.stdout)
        print("--- STDERR ---\n", proc.stderr)

    assert proc.returncode == 0, f"Example script failed (exit {proc.returncode})"
    assert "Starting tiny training run" in proc.stdout
