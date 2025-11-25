from biolm_utils.config import Config, get_config
from biolm_utils.plugin_registry import (
    apply_plugin,
    get_current_plugin,
    list_plugins,
    register_plugin,
    unregister_plugin,
)


def get_lab_plugin_config():
    """Mock plugin factory returning a config dict."""
    return {
        "model_cls_for_pretraining": None,
        "model_cls_for_finetuning": None,
        "tokenizer_cls": None,
        "learning_rate": 1e-3,
        "max_grad_norm": 1.0,
        "weight_decay": 0.0,
        "special_tokenizer_for_trainer_cls": None,
        "datacollator_cls_for_pretraining": None,
        "datacollator_cls_for_finetuning": None,
        "add_special_tokens": False,
        "config_cls": None,
        "pretraining_required": False,
        "dataset_cls": None,
    }


def test_lab_plugin_factory_returns_dict():
    cfg_dict = get_lab_plugin_config()
    assert isinstance(cfg_dict, dict)
    assert "learning_rate" in cfg_dict


def test_register_apply_unregister_behavior():
    # Register
    register_plugin("lab_test", get_lab_plugin_config)
    apply_plugin("lab_test")
    assert get_current_plugin() == "lab_test"
    # applied config should be a Config with matching learning_rate
    active = get_config()
    assert isinstance(active, Config)
    assert active.learning_rate == get_lab_plugin_config()["learning_rate"]

    # cleanup
    unregister_plugin("lab_test")
    # plugin must be gone
    assert "lab_test" not in list_plugins()
