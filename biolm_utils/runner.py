"""Provide the run-once worker used by CrossValidator.

This module contains a factory to create the run_once function used by the
CrossValidator. The returned function implements the core per-fold logic that
previously lived as a nested function inside biolm.main(). Keeping the same
signature keeps migration simple.
"""

from typing import Any, Callable, Optional

from transformers.data.data_collator import DefaultDataCollator

from biolm_utils.interpret import loo_scores
from biolm_utils.mlflow_integration import start_mlflow_run
from biolm_utils.params import get_detected_ngpus
from biolm_utils.train_tokenizer import tokenize
from biolm_utils.train_utils import (
    create_reports,
    get_model_and_config,
    get_tokenizer,
    get_trainer,
)


def make_run_fn(args, config, tokenizer, tokenizer_for_trainer, full_dataset):
    """Return a function implementing one training/validation/test run.

    Signature matches legacy code for compatibility:
      run(train_dataset, val_dataset, test_dataset, model_load_path, model_save_path, report_file, rank_file)
    """

    def run(
        train_dataset,
        val_dataset,
        test_dataset,
        model_load_path,
        model_save_path,
        report_file,
        rank_file,
    ):
        # Fast-path for tokenize mode: it doesn't use models and should return early.
        if args.mode == "tokenize":
            return tokenize(args)

        # When MLflow is enabled in the settings, start a run for this fold.
        with start_mlflow_run(model_save_path, args, config) as _mlflow:

            model_cls_map = {
                "pre-train": config.MODEL_CLS_FOR_PRETRAINING,
                "fine-tune": config.MODEL_CLS_FOR_FINETUNING,
                "predict": config.MODEL_CLS_FOR_FINETUNING,
                "interpret": config.MODEL_CLS_FOR_FINETUNING,
            }
            model_cls = model_cls_map.get(args.mode)
            if model_cls is None:
                raise ValueError(f"Unknown mode: '{args.mode}'.")

            if args.mode == "pre-train":
                data_collator = config.DATACOLLATOR_CLS_FOR_PRETRAINING(
                    tokenizer=tokenizer
                )
            else:
                data_collator = DefaultDataCollator()

            # The training path
            if args.mode in ["pre-train", "fine-tune"]:
                results, model = _train(
                    args,
                    train_dataset,
                    val_dataset,
                    data_collator,
                    model_load_path,
                    model_save_path,
                    tokenizer,
                    tokenizer_for_trainer,
                    full_dataset,
                    model_cls,
                    config,
                )

                if args.mode == "fine-tune" and test_dataset:
                    results = _test(
                        args,
                        test_dataset,
                        data_collator,
                        model_save_path,
                        report_file,
                        rank_file,
                        tokenizer,
                        tokenizer_for_trainer,
                        full_dataset,
                        model_cls,
                        config,
                        model,
                    )

                # If MLflow is active and results is a mapping, log numeric metrics
                if _mlflow is not None and isinstance(results, dict):
                    try:
                        numeric = {
                            k: float(v)
                            for k, v in results.items()
                            if isinstance(v, (int, float))
                        }
                        if numeric:
                            _mlflow.log_metrics(numeric)
                    except Exception:
                        pass

                return results

            elif args.mode == "predict":
                results = _test(
                    args,
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
                    None,
                )

                if _mlflow is not None and isinstance(results, dict):
                    try:
                        numeric = {
                            k: float(v)
                            for k, v in results.items()
                            if isinstance(v, (int, float))
                        }
                        if numeric:
                            _mlflow.log_metrics(numeric)
                    except Exception:
                        pass
                return results

            elif args.mode == "interpret":
                res = loo_scores(
                    args=args,
                    tokenizer=tokenizer,
                    model_cls=model_cls,
                    test_dataset=test_dataset,
                    model_load_path=model_load_path,
                    output_path=model_save_path,
                    remove_first_last=config.ADD_SPECIAL_TOKENS,
                )
                # interpretation outputs are not typically numeric metrics — return directly
                return res

        # tokenize handled early as fast path

    return run


def _train(
    args,
    train_dataset,
    val_dataset,
    data_collator,
    model_load_path,
    model_save_path,
    tokenizer,
    tokenizer_for_trainer,
    full_dataset,
    model_cls,
    config,
):
    # Import heavy dependencies lazily to keep tests fast
    from transformers.trainer_callback import TrainerState
    from transformers.training_args import TrainingArguments

    from biolm_utils.entry import (
        CHECKPOINTPATH,
        CLASSIFICATIONTRAINER_CLS,
        GRADACC,
        METRIC,
        MLMTRAINER_CLS,
        REGRESSIONTRAINER_CLS,
        TBPATH,
    )
    from biolm_utils.entry import args as global_args

    # delegate to existing helpers in train_utils for maintaining identical behavior
    return __import__("biolm_utils.biolm").biolm.train(
        train_dataset,
        val_dataset,
        data_collator,
        model_load_path,
        model_save_path,
        tokenizer,
        tokenizer_for_trainer,
        full_dataset,
        model_cls,
        config,
    )


def _test(
    args,
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
    return __import__("biolm_utils.biolm").biolm.test(
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
    )
