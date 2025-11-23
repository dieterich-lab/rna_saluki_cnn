"""Entry-point adapter for the Saluki plugin.

This module exposes a self-contained factory function used by the framework's
entry-point discovery. The function constructs the plugin `Config` using the
package-local dataset/model placeholders so the package remains fully
installable and functional during integration tests.
"""

from __future__ import annotations

from biolm_utils.config import Config

from .dataset import SalukiDataset
from .models import SalukiModel


def get_saluki_config() -> Config:
    # This factory is intentionally small and import-safe so it can be used
    # by the framework immediately after discovery without heavy runtime
    # effects.
    try:
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


__all__ = ["get_saluki_config"]
