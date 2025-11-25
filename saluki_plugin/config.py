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
        PluginConfig: The complete plugin configuration.
    """
    from biolm_utils.plugin_config import PluginConfig, PluginManager
    from transformers import DefaultDataCollator, PreTrainedTokenizerFast

    # Create the plugin configuration
    # Modify these settings for your custom plugin
    config = PluginConfig(
        # Model classes - set your model classes here
        model_cls_for_pretraining=None,  # Set if you have pretraining
        model_cls_for_finetuning=SalukiModel,  # Your main model class
        # Dataset class - your dataset implementation
        dataset_cls=RNACNNDataset,
        # Tokenizer - usually PreTrainedTokenizerFast
        tokenizer_cls=PreTrainedTokenizerFast,
        # Data collators - customize for your data preprocessing needs
        datacollator_cls_for_pretraining=None,  # For pretraining (if any)
        datacollator_cls_for_finetuning=DefaultDataCollator,  # For finetuning
        # Tokenizer settings
        add_special_tokens=False,
        # Model config class (optional)
        config_cls=None,
        # Training settings
        pretraining_required=False,
        learning_rate=1e-4,
        max_grad_norm=1.0,
        weight_decay=0.0,
        # Special tokenizer (optional)
        special_tokenizer_for_trainer_cls=None,
    )

    # Make this the active configuration in the framework
    PluginManager.set_config(config)

    return config


__all__ = ["get_saluki_config"]
