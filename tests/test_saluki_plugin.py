"""Tests ensuring the Saluki plugin is a thin wrapper delegating to biolm_utils."""

from biolm_utils.plugin_registry import get_plugin_factory, get_current_plugin, apply_plugin
from biolm_utils.config import get_config


def test_saluki_plugin_registered():
    # importing the plugin module registers the factory
    import saluki  # this registers the plugin factory on import

    factory = get_plugin_factory("saluki")
    assert factory is not None, "saluki plugin factory should be registered"


def test_saluki_apply_sets_config():
    # Ensure applying the plugin activates the plugin config inside the framework
    import saluki

    apply_plugin("saluki")
    cfg = get_config()
    assert cfg is not None
    # Expect the active plugin to be 'saluki'
    assert get_current_plugin() == "saluki"
    # The dataset class should match the SalukiDataset type by name
    assert hasattr(cfg, "DATASET_CLS") or hasattr(cfg, "dataset_cls")
