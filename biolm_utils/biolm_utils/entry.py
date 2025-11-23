import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from transformers.trainer import Trainer

from biolm_utils.params import get_detected_ngpus, load_config
from biolm_utils.train_utils import (
    compute_metrics_for_classification,
    compute_metrics_for_regression,
)
from biolm_utils.trainer import (
    RegressionTrainer,
    WeightedRegressionTrainer,
    WeightedSamplingTrainer,
)


# Get the arguments from the command line.
# Support legacy CLI usage (e.g., `saluki.py tokenize --filepath /x`) by
# translating common `--flag`/subcommand styles into Hydra-style overrides
# and passing them into `load_config(overrides=[...])` where present.
def _legacy_argv_to_overrides(argv: list) -> list:
    # If any arg already looks like `key=value`, assume Hydra-style and
    # don't perform legacy translation.
    if any(("=" in a) for a in argv[1:]):
        return []

    overrides = []
    # Recognize a positional mode (tokenize, fine-tune, predict, interpret)
    if len(argv) > 1 and not argv[1].startswith("-"):
        overrides.append(f"mode={argv[1]}")

    legacy_map = {
        "--filepath": "data_source.filepath",
        "--outputpath": "outputpath",
        "--dev": "debugging.dev",
        "--silent": "debugging.silent",
        "--encoding": "tokenization.encoding",
        "--seqpos": "data_source.seqpos",
        "--idpos": "data_source.idpos",
        "--labelpos": "data_source.labelpos",
        "--columnsep": "data_source.columnsep",
        "--stripheader": "data_source.stripheader",
        "--accelerator": "debugging.accelerator",
    }

    # iterate argv and detect flags
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in legacy_map:
            key = legacy_map[a]
            # boolean flags (no value) use true
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                overrides.append(f"{key}=true")
                i += 1
            else:
                val = argv[i + 1]
                # coerce tabs and other special chars safely
                if val == "\t":
                    val = "\\t"
                overrides.append(f"{key}={val}")
                i += 2
        else:
            i += 1

    return overrides


args = load_config(overrides=_legacy_argv_to_overrides(sys.argv))

# Structured-config compatibility helpers (prefer nested dataclass fields)
data_source = getattr(args, "data_source", None)
training = getattr(args, "training", None)
debugging = getattr(args, "debugging", None)
inference = getattr(args, "inference", None)


def t_get(key, default=None):
    if training is not None and hasattr(training, key):
        return getattr(training, key)
    return getattr(args, key, default)


def d_get(key, default=None):
    if debugging is not None and hasattr(debugging, key):
        return getattr(debugging, key)
    return getattr(args, key, default)


def i_get(key, default=None):
    if inference is not None and hasattr(inference, key):
        return getattr(inference, key)
    return getattr(args, key, default)


# Switch off the 'The used dataset had no length, returning gathered tensors. You should drop the remainder yourself.' warning if desired.
# if args.silent:
logging.getLogger("accelerate").setLevel(logging.WARNING)

if args.outputpath is None:
    if hasattr(args, "filepath") and args.filepath:
        args.outputpath = Path(args.filepath).stem
    else:
        args.outputpath = "output"

OUTPUTPATH = Path(args.outputpath)
OUTPUTPATH.mkdir(parents=True, exist_ok=True)

TOKENIZERFILE = OUTPUTPATH / "tokenizer.json"
from biolm_utils.params import get_detected_ngpus, load_config

if getattr(args, "mode", None) == "fine-tune":
    MODELLOADPATH = OUTPUTPATH / "pre-train"
elif args.mode in ["interpret", "predict"]:
    MODELLOADPATH = OUTPUTPATH / "fine-tune"
else:
    MODELLOADPATH = None  # not needed for pre-training

# `pretrainedmodel` changes either:
# - different tokenizer when pre-training
# - different pre-trained-model/tokenizer when fine-tuning
# - tokenizer/fine-tuned model path for inference
if i_get("pretrainedmodel", getattr(args, "pretrainedmodel", None)):
    if args.mode != "pre-train":
        MODELLOADPATH = Path(args.pretrainedmodel)
        TOKENIZERFILE = MODELLOADPATH / "tokenizer.json"
    else:
        TOKENIZERFILE = Path(args.pretrainedmodel) / "tokenizer.json"

# if not args.mode in ["predict", "interpret"]:
#     MODELSAVEPATH = OUTPUTPATH / args.mode
# else:
#     MODELSAVEPATH = None  # Not needed for inference tasks
# MODELSAVEPATH = OUTPUTPATH
MODELSAVEPATH = OUTPUTPATH / args.mode

if args.mode not in ["tokenize", "predict", "interpret"]:
    MODELSAVEPATH.mkdir(parents=True, exist_ok=True)
REPORTFILE = MODELSAVEPATH / "test_predictions.csv"
RANKFILE = MODELSAVEPATH / "rank_deltas.csv"
TBPATH = MODELSAVEPATH / "tboard"
LOGPATH = MODELSAVEPATH / "logs"
LOGPATH.mkdir(parents=True, exist_ok=True)
if args.mode not in ["tokenize", "predict", "interpret"]:
    TBPATH.mkdir(parents=True, exist_ok=True)

if args.mode in ["tokenize"]:
    DATASETFILE = None  # we don't save it when tokenizing
else:
    DATASETFILE = OUTPUTPATH / args.mode / "dataset.json"

# Set up logging
now = datetime.now().strftime("%Y-%m-%d_%H:%M")
LOGFILE = LOGPATH / f"{now}.log"
LOGFILE.touch(exist_ok=True)
if not d_get("dev", False):
    handlers = [
        logging.FileHandler(LOGFILE, mode="w"),
        logging.StreamHandler(),
    ]
else:
    handlers = [
        logging.StreamHandler(),
    ]

# Convert all handlers to logging.Handler if not already
handlers = [
    h if isinstance(h, logging.Handler) else logging.StreamHandler() for h in handlers
]

logging.basicConfig(
    format=f"%(asctime)s ({args.mode} {OUTPUTPATH.stem}) - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=handlers,
)

# We scale the gradient with respect to the number of GPUs to keep an
# effective batch size of `args.batchsize` x `args.gradacc`
if d_get("dev", False):
    GRADACC = 1
else:
    # Use the auto-detected GPU count (or 1 when CPU) instead of the removed `args.ngpus`.
    detected_gpus = get_detected_ngpus(args)
    GRADACC = float(t_get("gradacc", getattr(args, "gradacc", 1))) / max(
        1, int(detected_gpus)
    )
    logging.info(f"Set gradient accumulation to {GRADACC}.")

# Log the arguments.
logging.info(f"{'=== Params ===':>32}")
for k, v in sorted(vars(args).items()):
    logging.info(f"{k:>25} : {str(v):<25}")


if getattr(args, "resume", False) == True or (
    getattr(args, "training", None) and getattr(args.training, "resume", False) == True
):
    CHECKPOINTPATH = max(MODELSAVEPATH.glob("checkpoint*"), key=os.path.getmtime)
    logging.info(f"Pretrained model to resume from: {CHECKPOINTPATH}")
else:
    CHECKPOINTPATH = None

REGRESSIONTRAINER_CLS = (
    WeightedRegressionTrainer
    if getattr(
        getattr(args, "training", None),
        "weightedregression",
        getattr(args, "weightedregression", False),
    )
    else RegressionTrainer
)

CLASSIFICATIONTRAINER_CLS = WeightedSamplingTrainer

MLMTRAINER_CLS = Trainer

METRIC = (
    compute_metrics_for_classification
    if getattr(args, "task", None) == "classification"
    else compute_metrics_for_regression
)
