from __future__ import annotations

from typing import Any

from transformers import PreTrainedModel


class SalukiModel(PreTrainedModel):
    config_class = None

    def __init__(self, config: Any, *args, **kwargs):
        super().__init__(config)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("SalukiModel is a placeholder and not runnable")
