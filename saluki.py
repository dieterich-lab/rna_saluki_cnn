"""Saluki plugin entrypoint (thin wrapper)

This file intentionally stays minimal: it defines a factory returning a
`biolm_utils.config.Config` instance for the Saluki plugin and registers the
factory with the `biolm_utils.plugin_registry` API.

The heavy runtime logic lives in `biolm_utils` (framework). This file only
exposes the plugin (dataset/model/tokenizer choices and defaults) and a
convenience function to activate the plugin via `apply_plugin('saluki')`.
"""

from __future__ import annotations

from typing import Optional

from biolm_utils.config import Config
from biolm_utils.plugin_registry import register_plugin, apply_plugin

from rna_cnn_dataset import SalukiDataset
from rna_cnn_models import SalukiModel


def get_saluki_config() -> Config:
    """Factory used to create a Config for the Saluki plugin.

    Keep this function lightweight and importable from tests. Avoid side
    effects — the heavy orchestration is handled by `biolm_utils`.
    """

    # We intentionally keep model classes as simple placeholders. Real training
    # or inference code will be executed by the framework which expects the
    # classes to be available, not instantiated at registration time.
    try:
        # Import light-weight tokenizer/data-collator from transformers if
        # available. If not present in the environment, tests and registration
        # will still work because dataclasses accept None for optional types.
        from transformers import DefaultDataCollator, PreTrainedTokenizerFast
    except Exception:  # pragma: no cover - environment-dependent
        DefaultDataCollator = None
        PreTrainedTokenizerFast = None

    return Config(
        model_cls_for_pretraining=None,
        model_cls_for_finetuning=SalukiModel,
        tokenizer_cls=PreTrainedTokenizerFast,
        learning_rate=1e-4,
        max_grad_norm=1.0,
        weight_decay=0.0,
        special_tokenizer_for_trainer_cls=None,
        datacollator_cls_for_pretraining=None,
        datacollator_cls_for_finetuning=DefaultDataCollator,
        add_special_tokens=False,
        config_cls=None,
        pretraining_required=False,
        dataset_cls=SalukiDataset,
    )


# Register the plugin eagerly on import so tests and `apply_plugin('saluki')`
# calls work as expected.
from biolm_utils.plugin_registry import get_plugin_factory, unregister_plugin

# Register the plugin eagerly on import so tests and `apply_plugin('saluki')`
# calls work as expected. Guard against re-registration when the module is
# imported multiple times within the same Python process (tests re-import).
if get_plugin_factory("saluki") is None:
    register_plugin("saluki", get_saluki_config)
else:
    # If a previous registration exists we leave it in-place (tests may
    # re-import this module repeatedly). If you want to force a refresh,
    # uncomment the next two lines to unregister and re-register explicitly.
    # unregister_plugin("saluki")
    # register_plugin("saluki", get_saluki_config)
    pass


def activate():
    """Convenience helper — apply the `saluki` plugin so the Config becomes
    active in the framework's registry.

    This is helpful for tests or quick CLI wrappers that want to activate the
    plugin before invoking `biolm_utils` orchestration.
    """
    apply_plugin("saluki")


if __name__ == "__main__":
    # Minimal CLI usage: activate the plugin and print a short summary.
    activate()
    from biolm_utils.config import get_config

    cfg = get_config()
    print("Activated saluki plugin — dataset:", cfg.DATASET_CLS)
# (no extra CLI or heavy-run logic here — the plugin stays thin)
