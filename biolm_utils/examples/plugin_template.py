"""Example plugin template for educational use.

This demonstrates the minimal structure expected by a plugin that integrates
with biolm_utils: provide a factory function returning a `Config` object and
register it via the `plugin_registry` API.

Use this as a scaffold for students to create custom plugins that plug into
the main biolm_utils orchestration (tokenize / fine-tune / predict / interpret).
"""

from transformers import BertConfig, DefaultDataCollator, PreTrainedTokenizerFast

from biolm_utils.config import Config


def get_example_plugin_config():
    # Minimal example values — replace with real model/dataset classes.
    # Use explicit keyword names so this example is clear and robust.
    # Prefer the canonical `learning_rate` key for new plugins. Use named
    # arguments to keep code robust and easy to read.
    return Config(
        model_cls_for_pretraining=None,
        model_cls_for_finetuning=None,
        tokenizer_cls=PreTrainedTokenizerFast,
        learning_rate=1e-4,
        max_grad_norm=1.0,
        weight_decay=0.0,
        special_tokenizer_for_trainer_cls=None,
        datacollator_cls_for_pretraining=None,
        datacollator_cls_for_finetuning=DefaultDataCollator,
        add_special_tokens=False,
        config_cls=BertConfig,
        pretraining_required=False,
        dataset_cls=None,
    )
