import warnings

from biolm_utils.cfg_migration import (
    DEFAULT_K_FOLD,
    analyze_crossvalidation,
    migrate_crossvalidation,
)
from biolm_utils.structured_config import BioLMConfig, DataSourceConfig


def _make_params(**kwargs):
    # Build a BioLMConfig with the provided parameters stored under data_source
    return BioLMConfig(data_source=DataSourceConfig(**kwargs))


def test_analyze_true_without_splitpos():
    p = _make_params(crossvalidation=True, splitratio=[80, 20])
    notes = analyze_crossvalidation(p)
    assert any("ambiguous" in n.lower() or "recommend" in n.lower() for n in notes)


def test_migrate_true_with_splitratio_applies_default():
    p = _make_params(crossvalidation=True, splitratio=[80, 20])
    new_p, notes = migrate_crossvalidation(p, auto_apply=True)
    # new_p is a BioLMConfig with data_source updated
    assert new_p.data_source.crossvalidation == DEFAULT_K_FOLD
    assert notes and any(
        "recommend" in n.lower() or "convert" in n.lower() for n in notes
    )


def test_migrate_zero_to_false():
    p = _make_params(crossvalidation=0)
    new_p, notes = migrate_crossvalidation(p, auto_apply=True)
    assert new_p.data_source.crossvalidation is False
