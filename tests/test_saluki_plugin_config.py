"""
Saluki plugin configuration tests.

These tests verify that:
1. Saluki plugin loads correctly via entry points
2. Plugin configuration is complete and correct
3. Pre-training is correctly disabled (Saluki doesn't support it)
4. All required attributes exist
"""

import logging

import pytest

logging.basicConfig(level=logging.DEBUG)


def debug_log(msg):
    """Print debug message to stdout for test visibility."""
    print(msg)


def test_saluki_plugin_loading():
    """Test that Saluki plugin can be loaded via entry points."""
    debug_log("Starting Saluki plugin loading test")

    import importlib.metadata

    from biolm.plugin_config import PluginManager

    # Test loading Saluki plugin via entry point
    eps = importlib.metadata.entry_points(group="biolm.plugins")
    saluki_ep = next((ep for ep in eps if ep.name == "saluki"), None)

    assert saluki_ep is not None, "Saluki plugin not found in entry points"
    debug_log("✓ Saluki entry point found")

    # Load the config function
    saluki_config_fn = saluki_ep.load()
    saluki_config_fn()
    config = PluginManager.get_config()

    debug_log(f"✓ Saluki plugin loaded successfully")
    assert config.pretraining_required == False, "Saluki should not require pretraining"
    debug_log(f"✓ Saluki pretraining_required: {config.pretraining_required}")


def test_saluki_unsupported_pretraining():
    """Test that Saluki plugin config shows pretraining_required=False."""
    debug_log("Starting Saluki pre-train check")

    import importlib.metadata

    from biolm.plugin_config import PluginManager

    # Load Saluki plugin via entry point
    eps = importlib.metadata.entry_points(group="biolm.plugins")
    saluki_ep = next(ep for ep in eps if ep.name == "saluki")
    config_fn = saluki_ep.load()
    config_fn()
    config = PluginManager.get_config()

    # Verify Saluki doesn't support pre-training
    assert config.pretraining_required == False, "Saluki should not require pretraining"
    debug_log("✓ Saluki correctly has pretraining_required=False")

    assert (
        config.model_cls_for_pretraining is None
    ), "Saluki should not have a pretraining model class"
    debug_log("✓ Saluki correctly has model_cls_for_pretraining=None")


def test_saluki_plugin_config():
    """Test that Saluki plugin configuration is complete."""
    debug_log("Starting Saluki config validation test")

    import importlib.metadata

    from biolm.plugin_config import PluginManager

    # Load Saluki plugin via entry point
    eps = importlib.metadata.entry_points(group="biolm.plugins")
    saluki_ep = next(ep for ep in eps if ep.name == "saluki")
    config_fn = saluki_ep.load()
    config_fn()
    config = PluginManager.get_config()

    # Verify all expected attributes are present and correct
    assert (
        config.model_cls_for_pretraining is None
    ), "model_cls_for_pretraining should be None"
    debug_log("✓ model_cls_for_pretraining = None (correct)")

    assert (
        config.model_cls_for_finetuning is not None
    ), "model_cls_for_finetuning is None"
    debug_log("✓ model_cls_for_finetuning exists")

    assert config.dataset_cls is not None, "dataset_cls is None"
    debug_log("✓ dataset_cls exists")

    assert config.tokenizer_cls is not None, "tokenizer_cls is None"
    debug_log("✓ tokenizer_cls exists")

    assert (
        config.datacollator_cls_for_pretraining is None
    ), "datacollator_cls_for_pretraining should be None"
    debug_log("✓ datacollator_cls_for_pretraining = None (correct)")

    assert (
        config.datacollator_cls_for_finetuning is not None
    ), "datacollator_cls_for_finetuning is None"
    debug_log("✓ datacollator_cls_for_finetuning exists")

    assert config.add_special_tokens == False, "add_special_tokens should be False"
    debug_log("✓ add_special_tokens = False")

    assert config.pretraining_required == False, "pretraining_required should be False"
    debug_log("✓ pretraining_required = False")

    debug_log("✓ All Saluki config attributes validated successfully")


def test_saluki_model_is_callable():
    """Test that Saluki finetuning model class can be instantiated."""
    debug_log("Starting Saluki model callability test")

    import importlib.metadata

    from biolm.plugin_config import PluginManager

    # Load Saluki plugin
    eps = importlib.metadata.entry_points(group="biolm.plugins")
    saluki_ep = next(ep for ep in eps if ep.name == "saluki")
    config_fn = saluki_ep.load()
    config_fn()
    config = PluginManager.get_config()

    # Verify finetuning model class is callable (has __init__)
    assert hasattr(
        config.model_cls_for_finetuning, "__init__"
    ), "Finetuning model not callable"
    debug_log("✓ Finetuning model is callable")

    # Verify pretraining model is None (Saluki doesn't support it)
    assert config.model_cls_for_pretraining is None, "Pretraining model should be None"
    debug_log("✓ Pretraining model correctly set to None")
