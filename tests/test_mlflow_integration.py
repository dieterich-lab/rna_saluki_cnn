import sys
from pathlib import Path

from biolm_utils.structured_config import BioLMConfig, SettingsConfig


def make_fake_mlflow(recorder: dict):
    class FakeRun:
        pass

    class FakeMLflow:
        def set_tracking_uri(self, uri):
            recorder.setdefault("set_tracking_uri", []).append(uri)

        def set_experiment(self, name):
            recorder.setdefault("set_experiment", []).append(name)

        def start_run(self, **kwargs):
            recorder.setdefault("start_run", []).append(kwargs)
            return FakeRun()

        def log_params(self, params):
            recorder.setdefault("log_params", []).append(params)

        def log_metrics(self, metrics):
            recorder.setdefault("log_metrics", []).append(metrics)

        def log_artifacts(self, path):
            recorder.setdefault("log_artifacts", []).append(path)

        def end_run(self):
            recorder.setdefault("end_run", []).append(True)

    return FakeMLflow()


def test_start_mlflow_run_enabled(tmp_path, monkeypatch):
    recorder = {}
    fake_mlflow = make_fake_mlflow(recorder)
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    # import here to use the lazy importer in the module under test
    from biolm_utils.mlflow_integration import start_mlflow_run

    args = BioLMConfig(
        mode="fine-tune",
        task="regression",
        outputpath=None,
        settings=SettingsConfig(
            mlflow={
                "enabled": True,
                "experiment_name": "test-exp",
                "log_artifacts": True,
            }
        ),
    )
    # ensure the model save path exists (artifact logging will attempt to read it)
    d = tmp_path / "model_save"
    d.mkdir()

    class CfgObj:
        learning_rate = 0.001

    cfg = CfgObj()

    with start_mlflow_run(d, args, cfg) as ml:
        # inside the context our fake mlflow should have been started
        assert "start_run" in recorder
        assert recorder.get("set_experiment") == ["test-exp"]
        assert recorder.get("log_params")

    # after context exit we expect artifacts and end_run recorded
    assert recorder.get("log_artifacts")
    assert recorder.get("end_run")


def test_start_mlflow_run_disabled(monkeypatch):
    recorder = {}
    fake_mlflow = make_fake_mlflow(recorder)
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    from biolm_utils.mlflow_integration import start_mlflow_run

    args = BioLMConfig(
        mode="fine-tune", settings=SettingsConfig(mlflow={"enabled": False})
    )

    class CfgObj2:
        learning_rate = 0.001

    cfg = CfgObj2()

    with start_mlflow_run(None, args, cfg) as ml:
        # ml should be None when disabled
        assert ml is None

    # ensure no mlflow calls were recorded
    assert recorder == {}
