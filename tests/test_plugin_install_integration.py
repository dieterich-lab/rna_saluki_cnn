import subprocess
import sys
from pathlib import Path

from biolm_utils import plugin_loader
from biolm_utils.plugin_registry import get_plugin_factory, unregister_plugin


def test_install_and_discover_plugin(tmp_path):
    # Ensure clean state
    if get_plugin_factory("saluki") is not None:
        unregister_plugin("saluki")

    repo_root = Path(__file__).resolve().parents[1]
    pkg_dir = repo_root / "saluki_plugin"

    # Install the package in editable mode using the current interpreter
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(pkg_dir)])

    try:
        # Discover entrypoints and ensure the plugin factory is registered
        plugin_loader.discover_entrypoint_plugins()
        assert get_plugin_factory("saluki") is not None
    finally:
        # Cleanup: uninstall the installed package and unregister plugin
        subprocess.check_call(
            [sys.executable, "-m", "pip", "uninstall", "-y", "saluki-plugin"]
        )  # package name as in pyproject
        try:
            unregister_plugin("saluki")
        except Exception:
            pass
