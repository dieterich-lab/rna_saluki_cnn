"""High-signal contract checks for Saluki plugin configuration."""

import importlib.metadata

from biolm.plugin_config import PluginManager


def _load_saluki_config():
    """Load Saluki plugin via entry point and return active PluginConfig."""
    eps = importlib.metadata.entry_points(group="biolm.plugins")
    saluki_ep = next((ep for ep in eps if ep.name == "saluki"), None)
    assert saluki_ep is not None, "Saluki plugin not found in entry points"

    saluki_ep.load()()
    return PluginManager.get_config()


def test_saluki_plugin_contract():
    """Verify the minimal, critical plugin contract for Saluki."""
    config = _load_saluki_config()

    # Core classes required for runtime
    assert config.model_cls_for_finetuning is not None
    assert config.dataset_cls is not None

    # Saluki is CNN-only (no pretraining phase)
    assert config.pretraining_required is False
    assert config.model_cls_for_pretraining is None
