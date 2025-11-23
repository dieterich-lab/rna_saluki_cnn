import pytest

from biolm_utils.runner import make_run_fn
from biolm_utils.structured_config import BioLMConfig, DebuggingConfig, TrainingConfig


class DummyConfig:
    # minimal config container used by make_run_fn tests
    DATACOLLATOR_CLS_FOR_PRETRAINING = lambda tokenizer=None: object()
    MODEL_CLS_FOR_PRETRAINING = object
    MODEL_CLS_FOR_FINETUNING = object
    ADD_SPECIAL_TOKENS = False


def _make_args(mode="predict"):
    # minimal structured config used by the runner
    return BioLMConfig(
        mode=mode,
        training=TrainingConfig(batchsize=1, nepochs=1, resume=False),
        debugging=DebuggingConfig(dev=False, silent=True),
        task="classification",
    )


def test_make_run_fn_invalid_mode_raises():
    args = _make_args(mode="not-a-mode")
    run_fn = make_run_fn(args, DummyConfig(), None, None, None)

    with pytest.raises(ValueError):
        run_fn(None, None, None, None, None, None, None)


def test_make_run_fn_tokenize_mode(monkeypatch):
    # Ensure tokenize branch is called and its return value propagated
    called = {}

    def fake_tokenize(a):
        called["args"] = a
        return "TOK"

    monkeypatch.setattr("biolm_utils.runner.tokenize", fake_tokenize)

    args = _make_args(mode="tokenize")
    run_fn = make_run_fn(args, DummyConfig(), None, None, None)

    out = run_fn(None, None, None, None, None, None, None)
    assert out == "TOK"


def test_predict_delegates_to_biolm_test(monkeypatch):
    # Stub the biolm.test function to verify arguments are forwarded
    mod = __import__("biolm_utils.biolm").biolm

    def fake_test(
        test_dataset,
        data_collator,
        model_load_path,
        report_file,
        rank_file,
        tokenizer,
        tokenizer_for_trainer,
        full_dataset,
        model_cls,
        config,
        model,
    ):
        # return a sentinel metric
        return 0.42

    monkeypatch.setattr(mod, "test", fake_test)

    args = _make_args(mode="predict")
    run_fn = make_run_fn(args, DummyConfig(), None, None, None)

    # Call run with minimal required parameters for predict; model_save_path etc are not used
    val = run_fn(None, None, "test_ds", "mload", "msave", "report.json", "ranks.csv")
    assert val == 0.42


def test_fine_tune_triggers_train_and_then_test(monkeypatch):
    mod = __import__("biolm_utils.biolm").biolm

    class ModelLike:
        @staticmethod
        def parameters():
            return []

    def fake_train(*_args, **_kwargs):
        # simulate returning results and a model object
        return ({"eval_f1": 0.5}, ModelLike())

    def fake_test(*_args, **_kwargs):
        return 0.99

    monkeypatch.setattr(mod, "train", fake_train)
    monkeypatch.setattr(mod, "test", fake_test)

    args = _make_args(mode="fine-tune")
    run_fn = make_run_fn(args, DummyConfig(), None, None, None)

    res = run_fn(
        "train_ds", "val_ds", "test_ds", "mload", "msave", "report.json", "ranks.csv"
    )
    # fine-tune returns results from test when a test_dataset is present
    assert res == 0.99


def test_interpret_delegates_to_loo_scores(monkeypatch):
    # Ensure interpret mode delegates to loo_scores and returns value
    called = {}

    def fake_loo_scores(**kwargs):
        called.update(kwargs)
        return {"some": "score"}

    monkeypatch.setattr("biolm_utils.runner.loo_scores", fake_loo_scores)

    args = _make_args(mode="interpret")
    run_fn = make_run_fn(args, DummyConfig(), None, None, None)

    out = run_fn(None, None, "test_ds", "load", "save", "report.csv", "ranks.csv")
    assert isinstance(out, dict)
    assert "some" in out
