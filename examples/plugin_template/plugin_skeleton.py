"""Minimal example plugin skeleton for educational use.

This file demonstrates the recommended pattern: a side-effect-free factory
function that returns a `biolm_utils.config.Config` instance and a small
registration call. Tests (and a thin wrapper) can import this module to register
and activate the plugin without performing heavy runtime steps.
"""

from biolm_utils.config import Config


def get_plugin_config():
    # Keep the example tiny — real plugins will supply dataset and model classes
    return Config(
        model_cls_for_pretraining=None,
        model_cls_for_finetuning=None,
        tokenizer_cls=None,
        learning_rate=1e-4,
        max_grad_norm=1.0,
        weight_decay=0.0,
        special_tokenizer_for_trainer_cls=None,
        datacollator_cls_for_pretraining=None,
        datacollator_cls_for_finetuning=None,
        add_special_tokens=False,
        config_cls=None,
        pretraining_required=False,
        dataset_cls=None,
    )


def register(registry):
    """Helper used by tests to register the example into a registry.

    This helper exists so tests can import a self-contained unit rather than
    depending on top-level side-effects.
    """
    registry.register_plugin("plugin_template_example", get_plugin_config)
