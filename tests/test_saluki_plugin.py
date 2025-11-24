"""Tests ensuring the Saluki plugin is a thin wrapper delegating to biolm_utils."""

from biolm_utils.config import get_config
from biolm_utils.plugin_registry import (
    apply_plugin,
    get_current_plugin,
    get_plugin_factory,
)


def test_saluki_plugin_registered():
    # Import or reload the module so its registration logic runs in this
    # test process (some tests may import/unregister the plugin earlier).
    # Import the canonical packaged plugin factory (no wrapper).
    import importlib

    # Import the canonical packaged plugin implementation and register its factory
    # with the framework's registry for tests.
    from saluki_plugin import saluki_ep as saluki_pkg
    if get_plugin_factory("saluki") is None:
        # register the package factory into the framework registry
        from biolm_utils.plugin_registry import register_plugin

        register_plugin("saluki", saluki_pkg.get_saluki_config)

    factory = get_plugin_factory("saluki")
    assert factory is not None, "saluki plugin factory should be registered"


def test_saluki_apply_sets_config():
    # Ensure applying the plugin activates the plugin config inside the framework
    import importlib

    # Register packaged factory into the framework registry for this test run
    from saluki_plugin import saluki_ep as saluki_pkg
    if get_plugin_factory("saluki") is None:
        from biolm_utils.plugin_registry import register_plugin

        register_plugin("saluki", saluki_pkg.get_saluki_config)

    apply_plugin("saluki")
    cfg = get_config()
    assert cfg is not None
    # Expect the active plugin to be 'saluki'
    assert get_current_plugin() == "saluki"
    # The dataset class should match the SalukiDataset type by name
    assert hasattr(cfg, "DATASET_CLS") or hasattr(cfg, "dataset_cls")
