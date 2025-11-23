"""Lightweight, opt-in MLflow integration helpers.

This module keeps the MLflow dependency optional. When enabled via the
Hydra config (settings.mlflow.enabled) we will lazily import MLflow and
start a run for each fold executed by the runner.

The behaviour is intentionally minimal: log params at run start, metrics if
the run returns a mapping, and optionally log some artifacts (model dir,
report file) when requested.
"""

from __future__ import annotations

import contextlib
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from biolm_utils.config import get_config


class MLflowNotInstalled(Exception):
    pass


def _import_mlflow():
    try:
        mlflow = importlib.import_module("mlflow")
        return mlflow
    except Exception as e:  # pragma: no cover - import error path
        raise MLflowNotInstalled(
            "MLflow is not installed. Install optional dependency 'mlflow' or disable settings.mlflow.enabled"
        ) from e


def _params_from_config(config_obj: Any) -> Dict[str, Any]:
    # Extract thin mapping of config dataclass fields to primitive types
    params: Dict[str, Any] = {}
    try:
        # dataclass-like objects expose __dict__ or dataclass fields
        if hasattr(config_obj, "__dict__"):
            for k, v in vars(config_obj).items():
                params[k] = v
    except Exception:
        # best-effort fallback
        params["config_repr"] = str(config_obj)
    return params


@contextlib.contextmanager
def start_mlflow_run(
    model_save_path: Optional[Path],
    args: Any,
    config: Any,
    override: Optional[Dict[str, Any]] = None,
):
    """Context manager to start an MLflow run if enabled in args.settings.mlflow.

    Yields the mlflow module when active (so callers can optionally log extra
    artifacts). When not enabled this yields None.
    """
    # Resilient path: `args` may be a `BioLMConfig` dataclass produced by the
    # loader or another plain object with the required attributes — accept both
    # call sites.
    mlflow_conf = None
    try:
        mlflow_conf = (
            getattr(args, "settings", None).mlflow
            if getattr(args, "settings", None)
            else None
        )
    except Exception:
        mlflow_conf = None

    if not mlflow_conf or not mlflow_conf.get("enabled", False):
        # Not enabled -> context does nothing
        yield None
        return

    mlflow = _import_mlflow()

    # configure tracking uri if set
    tracking_uri = mlflow_conf.get("tracking_uri")
    if tracking_uri:
        try:
            mlflow.set_tracking_uri(tracking_uri)
        except Exception:
            # ignore errors when setting tracking uri — caller should handle
            pass

    experiment_name = mlflow_conf.get("experiment_name") or f"biolm_utils-{args.mode}"
    mlflow.set_experiment(experiment_name)

    run_name_template = mlflow_conf.get("run_name_template")
    name = None
    if run_name_template:
        try:
            name = run_name_template.format(mode=args.mode, path=str(model_save_path))
        except Exception:
            name = None

    run_kwargs: Dict[str, Any] = {}
    if name:
        run_kwargs["run_name"] = name

    # Start run
    run = mlflow.start_run(**run_kwargs)

    # Log basic params
    params = _params_from_config(config)
    # include some args (mode, task and other top-level simple values) — best effort
    extra = {}
    for key in ("mode", "task", "outputpath"):
        if hasattr(args, key):
            extra[key] = getattr(args, key)
    params.update(extra)
    if override:
        params.update(override)

    try:
        # mlflow.log_params expects simple values; convert complex types to json
        safe_params = {}
        for k, v in params.items():
            try:
                json.dumps(v)
                safe_params[k] = v
            except Exception:
                safe_params[k] = str(v)
        if safe_params:
            mlflow.log_params(safe_params)
    except Exception:
        # tolerate logging failures
        pass

    try:
        yield mlflow

        # At the end of the run we may want to log metrics/artifacts. The
        # caller should log metrics explicitly, but as a convenience we allow
        # the caller to rely on default behavior.
    finally:
        # Optionally log artifacts (model files, reports, tokenizer) if configured
        try:
            if mlflow_conf.get("log_artifacts", True):
                # model_save_path may point to a directory with weights — best-effort
                if model_save_path and model_save_path.exists():
                    try:
                        mlflow.log_artifacts(str(model_save_path))
                    except Exception:
                        # some tracking backends may not support artifact upload for dirs
                        pass
        except Exception:
            pass

        try:
            mlflow.end_run()
        except Exception:
            pass
