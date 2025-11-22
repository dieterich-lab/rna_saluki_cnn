import sys

# Minimal help quick-exit placed before heavy imports so `python saluki.py --help`
# behaves as a lightweight CLI help producer without importing large libs.
if "--help" in sys.argv or "-h" in sys.argv:
    print("usage: saluki.py <mode> [--help] [--filepath PATH] [other options]")
    sys.exit(0)

from transformers import BertConfig, DefaultDataCollator, PreTrainedTokenizerFast
from transformers.image_processing_utils import BaseImageProcessor

from biolm_utils.config import Config
from biolm_utils.plugin_registry import apply_plugin, register_plugin
from rna_cnn_dataset import RNACNNDataset
from rna_cnn_models import HFSaluki


def get_saluki_config() -> Config:
    """Return a clear, self-documenting Config object for the Saluki plugin.

    Using named kwargs makes plugin configuration readable and robust against
    changes to the `Config` dataclass ordering.
    """

    return Config(
        model_cls_for_pretraining=None,
        model_cls_for_finetuning=HFSaluki,
        tokenizer_cls=PreTrainedTokenizerFast,
        learning_rate=1e-3,
        max_grad_norm=0.4,
        weight_decay=0.001,
        special_tokenizer_for_trainer_cls=BaseImageProcessor,
        datacollator_cls_for_pretraining=None,
        datacollator_cls_for_finetuning=DefaultDataCollator,
        add_special_tokens=False,
        config_cls=BertConfig,
        pretraining_required=False,
        dataset_cls=RNACNNDataset,
    )


# Set the plugin config in the shared biolm_utils config object. Tests and the
# project entrypoint import this file and call `main()` below, so this ensures
# the Saluki plugin is registered programmatically with a readable API.
register_plugin("saluki", get_saluki_config)
# Make this plugin the active config for tests and CLI entry
apply_plugin("saluki")

from biolm_utils.biolm import main

main()
