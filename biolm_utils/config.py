from dataclasses import dataclass

from torch.utils.data import Dataset
from transformers import (
    DefaultDataCollator,
    PretrainedConfig,
    PreTrainedModel,
    XLNetTokenizerFast,
)
from transformers.image_processing_utils import ImageProcessingMixin


@dataclass
class Config:
    # Use snake_case names internally; keep the same field order so
    # positional construction (Config(*params)) remains compatible.
    model_cls_for_pretraining: PreTrainedModel  # 0
    model_cls_for_finetuning: PreTrainedModel  # 1
    tokenizer_cls: XLNetTokenizerFast  # 2
    # canonical snake_case name; keep position so positional construction
    # remains compatible with legacy callers which used the shorter name.
    learning_rate: float  # 4  (canonical snake_case)
    max_grad_norm: float  # 5
    weight_decay: float  # 6
    special_tokenizer_for_trainer_cls: ImageProcessingMixin  # 7
    datacollator_cls_for_pretraining: DefaultDataCollator  # 8
    datacollator_cls_for_finetuning: DefaultDataCollator  # 9
    add_special_tokens: bool  # 10
    config_cls: PretrainedConfig  # 11
    pretraining_required: bool  # 12
    dataset_cls: Dataset  # 13

    # Backwards compatible property names. This lets existing code that used
    # `Config.MODEL_CLS_FOR_PRETRAINING` keep working while the internal
    # representation is now snake_case. We do not modify attribute storage —
    # these properties simply mirror the values.
    def __init__(self, *args, **kwargs):
        # Support both positional and keyword construction, and accept
        # legacy UPPERCASE keyword names. Maintain the same field order as
        # before so positional construction remains compatible.
        field_names = [
            "model_cls_for_pretraining",
            "model_cls_for_finetuning",
            "tokenizer_cls",
            "learning_rate",
            "max_grad_norm",
            "weight_decay",
            "special_tokenizer_for_trainer_cls",
            "datacollator_cls_for_pretraining",
            "datacollator_cls_for_finetuning",
            "add_special_tokens",
            "config_cls",
            "pretraining_required",
            "dataset_cls",
        ]

        mapping = {
            "MODEL_CLS_FOR_PRETRAINING": "model_cls_for_pretraining",
            "MODEL_CLS_FOR_FINETUNING": "model_cls_for_finetuning",
            "TOKENIZER_CLS": "tokenizer_cls",
            "MAX_GRAD_NORM": "max_grad_norm",
            "WEIGHT_DECAY": "weight_decay",
            "SPECIAL_TOKENIZER_FOR_TRAINER_CLS": "special_tokenizer_for_trainer_cls",
            "DATACOLLATOR_CLS_FOR_PRETRAINING": "datacollator_cls_for_pretraining",
            "DATACOLLATOR_CLS_FOR_FINETUNING": "datacollator_cls_for_finetuning",
            "ADD_SPECIAL_TOKENS": "add_special_tokens",
            "CONFIG_CLS": "config_cls",
            "PRETRAINING_REQUIRED": "pretraining_required",
            "DATASET_CLS": "dataset_cls",
        }

        values: dict = {}

        # positional args (by original field order)
        for name, val in zip(field_names, args):
            values[name] = val

        # map provided kwargs (normalize uppercase legacy names)
        for k, v in kwargs.items():
            norm = mapping.get(k, k)
            values[norm] = v

        # ensure every field is present
        missing = [n for n in field_names if n not in values]
        if missing:
            raise TypeError(f"Missing fields for Config: {missing}")

        # Set attributes
        for n in field_names:
            object.__setattr__(self, n, values[n])

    @property
    def MODEL_CLS_FOR_PRETRAINING(self):
        return self.model_cls_for_pretraining

    @property
    def MODEL_CLS_FOR_FINETUNING(self):
        return self.model_cls_for_finetuning

    @property
    def TOKENIZER_CLS(self):
        return self.tokenizer_cls

    # Breaking change: removed backward-compatible uppercase accessors for
    # learning_rate. Use `config.learning_rate` instead.

    @property
    def MAX_GRAD_NORM(self):
        return self.max_grad_norm

    @property
    def WEIGHT_DECAY(self):
        return self.weight_decay

    @property
    def SPECIAL_TOKENIZER_FOR_TRAINER_CLS(self):
        return self.special_tokenizer_for_trainer_cls

    @property
    def DATACOLLATOR_CLS_FOR_PRETRAINING(self):
        return self.datacollator_cls_for_pretraining

    @property
    def DATACOLLATOR_CLS_FOR_FINETUNING(self):
        return self.datacollator_cls_for_finetuning

    @property
    def ADD_SPECIAL_TOKENS(self):
        return self.add_special_tokens

    @property
    def CONFIG_CLS(self):
        return self.config_cls

    @property
    def PRETRAINING_REQUIRED(self):
        return self.pretraining_required

    @property
    def DATASET_CLS(self):
        return self.dataset_cls


_config: Config | None = None


def get_config():
    global _config

    if _config is None:
        raise Exception("Config not initialized")
    return _config


def set_config(config: Config):
    global _config
    # Keep accepting the dataclass constructed in either legacy positional
    # style or using keyword args. If callers pass a mapping-like object we
    # try to convert it.
    if isinstance(config, Config):
        _config = config
        return

    # If config looks like a mapping or an object with legacy uppercase
    # attributes, normalize it into the new Config dataclass.
    try:
        # Accept mapping-like objects
        if hasattr(config, "items"):
            data = dict(config.items())
        else:
            # Attempt to create a dictionary from attributes
            data = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    except Exception:
        raise TypeError("Unrecognised config object passed to set_config")

    # Map legacy uppercase keys to snake_case names when present.
    mapping = {
        "MODEL_CLS_FOR_PRETRAINING": "model_cls_for_pretraining",
        "MODEL_CLS_FOR_FINETUNING": "model_cls_for_finetuning",
        "TOKENIZER_CLS": "tokenizer_cls",
        # NOTE: compatibility for legacy uppercase keys for the learning_rate
        # field has been removed as a deliberate breaking change. The mapping
        # intentionally does not include uppercase variants for this key.
        "MAX_GRAD_NORM": "max_grad_norm",
        "WEIGHT_DECAY": "weight_decay",
        "SPECIAL_TOKENIZER_FOR_TRAINER_CLS": "special_tokenizer_for_trainer_cls",
        "DATACOLLATOR_CLS_FOR_PRETRAINING": "datacollator_cls_for_pretraining",
        "DATACOLLATOR_CLS_FOR_FINETUNING": "datacollator_cls_for_finetuning",
        "ADD_SPECIAL_TOKENS": "add_special_tokens",
        "CONFIG_CLS": "config_cls",
        "PRETRAINING_REQUIRED": "pretraining_required",
        "DATASET_CLS": "dataset_cls",
    }

    normalized = {}
    for k, v in data.items():
        mapped = mapping.get(k, k)
        normalized[mapped] = v

    # Build Config instance from normalized dict. Allow missing keys to raise
    # a clear error to the caller.
    try:
        _config = Config(**normalized)
    except TypeError as e:
        raise TypeError(f"Failed to convert legacy config to new Config: {e}") from e
