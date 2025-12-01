import pytest
from biolm.cross_validation import CrossValidator
from biolm.paths import Paths
from biolm.structured_config import (
    BioLMConfig,
    DataSourceConfig,
    DebuggingConfig,
    TrainingConfig,
)


class DummyDataset:
    def __init__(self, n=10):
        self.n = n
        self.lines = ["1"] * n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"labels": 0}


def _base_paths(tmp_path):
    # simple base paths for tests
    base = tmp_path / "out"
    base.mkdir()
    return Paths(
        model_load_path=None,
        model_save_path=base,
        output_path=tmp_path,
        report_file=base / "preds.csv",
        rank_file=base / "ranks.csv",
    )


def test_crossval_random(tmp_path):
    params = BioLMConfig(
        mode="fine-tune",
        data_source=DataSourceConfig(crossvalidation=2, splitratio=[50, 50]),
        training=TrainingConfig(batchsize=1),
        debugging=DebuggingConfig(dev=False),
    )
    data = DummyDataset(n=8)
    results = []

    def run_once(train, val, test, model_load, model_save, report, rank):
        results.append((len(train), len(val), 1 if test is not None else 0))
        return len(train)

    cv = CrossValidator(
        params=params,
        dataset=data,
        run_once_fn=run_once,
        base_paths=_base_paths(tmp_path),
    )
    out = cv.execute()
    assert isinstance(out, list)
    assert len(out) == 2


def test_crossval_predict(tmp_path):
    params = BioLMConfig(
        mode="predict",
        data_source=DataSourceConfig(inferenceonsplits=None, columnsep=",", splitpos=1),
        debugging=DebuggingConfig(dev=False),
    )
    data = DummyDataset(n=5)
    calls = []

    def run_once(train, val, test, model_load, model_save, report, rank):
        calls.append((train, val, test))
        return True

    cv = CrossValidator(
        params=params,
        dataset=data,
        run_once_fn=run_once,
        base_paths=_base_paths(tmp_path),
    )
    out = cv.execute()
    assert out is True
    assert len(calls) == 1
    assert calls[0][0] is None and calls[0][1] is None and calls[0][2] is not None


def test_compat_parametrized_decorator(tmp_path):
    # Validate that the legacy decorator wrapper path still functions.
    from biolm.cross_validation import parametrized_decorator

    params = BioLMConfig(
        mode="predict",
        data_source=DataSourceConfig(inferenceonsplits=None, columnsep=",", splitpos=1),
        debugging=DebuggingConfig(dev=False),
    )
    data = DummyDataset(n=3)

    called = []

    def run_once(train, val, test, model_load, model_save, report, rank):
        called.append((train, val, test))
        return True

    decorated = parametrized_decorator(params, data)(run_once)
    out = decorated()
    assert out is True
    assert len(called) == 1


def test_cv_true_without_splitpos_raises(tmp_path):
    params = BioLMConfig(
        mode="fine-tune",
        data_source=DataSourceConfig(crossvalidation=True),
        debugging=DebuggingConfig(dev=False),
    )
    data = DummyDataset(n=6)

    def run_once(*_a, **_k):
        return True

    with pytest.raises(ValueError):
        CrossValidator(
            params=params,
            dataset=data,
            run_once_fn=run_once,
            base_paths=_base_paths(tmp_path),
        )


def test_cv_int_without_splitratio_raises(tmp_path):
    params = BioLMConfig(
        mode="fine-tune",
        data_source=DataSourceConfig(crossvalidation=3),
        debugging=DebuggingConfig(dev=False),
    )
    data = DummyDataset(n=12)

    def run_once(*_a, **_k):
        return True

    with pytest.raises(ValueError):
        CrossValidator(
            params=params,
            dataset=data,
            run_once_fn=run_once,
            base_paths=_base_paths(tmp_path),
        )


def test_cv_int_with_splitpos_conflict_raises(tmp_path):
    params = BioLMConfig(
        mode="fine-tune",
        data_source=DataSourceConfig(
            crossvalidation=3, splitpos=1, splitratio=[80, 20]
        ),
        debugging=DebuggingConfig(dev=False),
    )
    data = DummyDataset(n=12)

    def run_once(*_a, **_k):
        return True

    with pytest.raises(ValueError):
        CrossValidator(
            params=params,
            dataset=data,
            run_once_fn=run_once,
            base_paths=_base_paths(tmp_path),
        )


def test_splitpos_without_devsplits_raises_no_cv(tmp_path):
    params = BioLMConfig(
        mode="fine-tune",
        data_source=DataSourceConfig(crossvalidation=0, splitpos=1),
        debugging=DebuggingConfig(dev=False),
    )
    data = DummyDataset(n=8)

    def run_once(*_a, **_k):
        return True

    with pytest.raises(ValueError):
        CrossValidator(
            params=params,
            dataset=data,
            run_once_fn=run_once,
            base_paths=_base_paths(tmp_path),
        )
