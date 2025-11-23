from biolm_utils.plugin_registry import get_plugin_factory, unregister_plugin


def test_plugin_template_registers():
    # Import the template and use its helper to register in the registry
    from examples.plugin_template import plugin_skeleton as skeleton

    # ensure idempotent behavior in case previous tests left the registry dirty
    if get_plugin_factory("plugin_template_example") is not None:
        unregister_plugin("plugin_template_example")

    skeleton.register(__import__("biolm_utils").plugin_registry)

    assert get_plugin_factory("plugin_template_example") is not None
