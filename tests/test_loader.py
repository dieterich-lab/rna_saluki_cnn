import pytest
from biolm import loader
from biolm.structured_config import BioLMConfig
from omegaconf import OmegaConf


def test_process_hydra_config_from_dictconfig():
    cfg = OmegaConf.create({"mode": "tokenize", "debugging": {"accelerator": "cpu"}})
    out = loader._process_hydra_config(cfg)
    assert isinstance(out, BioLMConfig)
    assert out.mode == "tokenize"
    assert out.debugging.accelerator == "cpu"


def test_load_config_overrides_accepts_list():
    # Ensure programmatic overrides are accepted and applied
    out = loader.load_config(["mode=tokenize", "debugging.accelerator=cpu"])
    assert isinstance(out, BioLMConfig)
    assert out.mode == "tokenize"
    assert out.debugging.accelerator == "cpu"


def test_load_config_rejects_legacy_ngpus_override():
    # Legacy overrides like settings.environment.ngpus are disallowed and should
    # raise a ValueError with a helpful message.
    with pytest.raises(ValueError):
        loader.load_config(["mode=tokenize", "settings.environment.ngpus=3"])
