from biolm_utils.config import Config
from biolm_utils.plugin_registry import (
    apply_plugin,
    get_current_plugin,
    list_plugins,
    register_plugin,
    restore_previous_plugin,
    unregister_plugin,
)


def test_register_and_apply():
    def factory():
        # return a minimal but valid Config constructed positionally
        return Config(
            None, None, None, 1e-4, 0.5, 0.0, None, None, None, False, None, False, None
        )

    name = "_test_plugin"
    try:
        register_plugin(name, factory)
        assert name in list_plugins()
        apply_plugin(name)
        assert get_current_plugin() == name
        # restore previous (pop the applied stack) — should still succeed
        restore_previous_plugin()
        # unregister the plugin
        unregister_plugin(name)
        assert name not in list_plugins()
    finally:
        # registry is intentionally minimal - tests can leave entries around
        pass
