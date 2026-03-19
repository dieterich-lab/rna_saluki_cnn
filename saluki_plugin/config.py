"""Plugin configuration for the Saluki model.

This module defines the complete configuration for the Saluki plugin.
Modify the PluginConfig below to customize the plugin for your needs.
"""

from .dataset import RNACNNDataset
from .models import SalukiModel


def get_saluki_config():
    """Factory function that creates and returns the plugin configuration.

    The framework calls this function when the plugin is loaded.
    It creates a PluginConfig, sets it as active, and returns it.

    Returns:
        tuple: (PluginConfig, dict) where dict contains plugin-specific Hydra defaults.
    """
    from biolm.plugin_config import PluginConfig, PluginManager
    from transformers import (
        DefaultDataCollator,
        PretrainedConfig,
        PreTrainedTokenizerFast,
    )

    # Create the plugin configuration
    # Modify these settings for your custom plugin
    config = PluginConfig(
        model_cls_for_pretraining=None,
        model_cls_for_finetuning=SalukiModel,
        dataset_cls=RNACNNDataset,
        tokenizer_cls=PreTrainedTokenizerFast,
        datacollator_cls_for_pretraining=None,
        datacollator_cls_for_finetuning=DefaultDataCollator,
        add_special_tokens=False,
        config_cls=PretrainedConfig,  # Set a valid config class
        pretraining_required=False,
        learning_rate=0.001,
        max_grad_norm=0.4,
        weight_decay=0.001,
        special_tokenizer_for_trainer_cls=None,
    )

    # Ensure fallback to PretrainedConfig
    config.config_cls = config.config_cls or PretrainedConfig

    # Make this the active configuration in the framework
    PluginManager.set_config(config)

    # Provide plugin-specific Hydra defaults
    defaults = {
        "tokenization": {
            "encoding": "atomic",  # Saluki requires atomic encoding
        },
        "training": {
            "blocksize": 12288,  # Saluki's fixed block size
        },
    }

    return config, defaults


__all__ = ["get_saluki_config"]
