from biolm_utils.config import Config


def get_lab_plugin_config() -> Config:
    """Return a minimal example Config used for tests/education.

    The Config dataclass requires a number of attributes; tests only assert
    that the returned object is an instance of Config and that it has a
    learning_rate attribute. We'll construct a minimal valid Config instance
    with placeholder values.
    """

    cfg = Config(
        model_cls_for_pretraining=None,
        model_cls_for_finetuning=None,
        tokenizer_cls=None,
        learning_rate=0.001,
        max_grad_norm=1.0,
        weight_decay=0.0,
        special_tokenizer_for_trainer_cls=None,
        datacollator_cls_for_pretraining=None,
        datacollator_cls_for_finetuning=None,
        add_special_tokens=True,
        config_cls=None,
        pretraining_required=False,
        dataset_cls=None,
    )

    return cfg
