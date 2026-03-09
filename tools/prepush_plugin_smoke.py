#!/usr/bin/env python3
"""Pre-push smoke checks for Saluki plugin repository."""

from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str], quiet: bool = False) -> None:
    kwargs = {}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    # Framework must be importable because Saluki contract tests use PluginManager.
    check = subprocess.run(
        ["poetry", "run", "python", "-c", "import biolm"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if check.returncode != 0:
        print(
            "pre-push check requires biolm to be installed in this Poetry environment.\n"
            "Install it once, e.g.: poetry run pip install -e /prj/RNA_NLP/biolm_utils"
        )
        raise SystemExit(1)

    # Ensure this plugin contributes entry-point metadata in the active env.
    run(["poetry", "run", "pip", "install", "-e", ".", "--no-deps"], quiet=True)

    # Run high-signal plugin smoke tests.
    run(
        [
            "poetry",
            "run",
            "pytest",
            "tests/test_saluki_plugin.py",
            "tests/test_saluki_plugin_config.py",
            "-q",
        ]
    )


if __name__ == "__main__":
    main()
