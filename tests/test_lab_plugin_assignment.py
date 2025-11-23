from biolm_utils.config import Config, get_config
from biolm_utils.plugin_registry import (
    apply_plugin,
    get_current_plugin,
    list_plugins,
    register_plugin,
    unregister_plugin,
)
from examples.lab_plugin_assignment.plugin_skeleton import get_lab_plugin_config


def test_lab_plugin_factory_returns_Config():
    cfg = get_lab_plugin_config()
    assert isinstance(cfg, Config)
    assert hasattr(cfg, "learning_rate")


def test_register_apply_unregister_behavior():
    # Register
    register_plugin("lab_test", get_lab_plugin_config)
    apply_plugin("lab_test")
    assert get_current_plugin() == "lab_test"
    # applied config should be a Config with matching learning_rate
    active = get_config()
    assert isinstance(active, Config)
    assert active.learning_rate == get_lab_plugin_config().learning_rate

    # cleanup
    unregister_plugin("lab_test")
    # plugin must be gone
    assert "lab_test" not in list_plugins()
