# bioml_utils — utilities for bioinformatic language models

## Overview

This repository provides the Saluki plugin for `biolm_utils`, a framework for tokenizing, pre-training, and fine-tuning language models on biological sequences.

The Saluki plugin includes a CNN-based model (`HFSaluki`) and dataset (`RNACNNDataset`) for RNA sequence analysis.

## Installation

Follow the single-env workflow to install both the framework and plugin:

```bash
# 1) Install framework
cd /path/to/biolm_utils
poetry install

# 2) Install plugin in same environment
cd /path/to/rna_saluki_cnn
poetry install
```

## Plugin Structure

The Saluki plugin follows the modern 3-file plugin architecture:

- **`saluki_plugin/dataset.py`** - RNA dataset implementation (`RNACNNDataset`)
- **`saluki_plugin/models.py`** - CNN model implementation (`SalukiModel`)  
- **`saluki_plugin/config.py`** - Plugin configuration factory (`get_saluki_config()`)

The `config.py` uses the new `PluginConfig` system with comprehensive documentation for each setting.

## Using Saluki

After installation, activate the plugin in your code:

```python
from biolm_utils.plugin_registry import apply_plugin
apply_plugin('saluki')  # Sets Saluki as active config
```

Then use the framework's CLI or API with Saluki's model and dataset.

## Quick start (Poetry)

This repo uses Poetry for reproducible installs. From the project root:

```bash
# optional: choose a Python interpreter
poetry env use $(which python)
poetry install

# run tests
poetry run pytest -q
```

To add MLflow extras for experiment tracking:

```bash
poetry install --with mlflow
```

## Pipfile → Poetry

This project migrated from Pipenv to Poetry. To migrate older environments:

```bash
poetry init --no-interaction
poetry lock
```

## Layout

Top-level package: `biolm_utils/` — main modules include:
- `biolm.py`     : CLI entrypoint for tokenize / pre-train / fine-tune / interpret / predict
- `config.py`    : Config dataclass and compatibility helpers
- `cross_validation.py` : New CrossValidator orchestration (replaces decorator-based CV)
- `params.py` / `entry.py` : CLI parsing and runtime wiring
- `train_tokenizer.py`, `trainer.py`, `interpret.py`, `loo_utils.py` : core functionality

See the `biolm_utils/` package for full details.


## Docs

See DOCS/ for a short guide on framework internals and a plugin example (Saluki).

## Plugin separation (recommended)

The recommended layout is to keep the *framework* (biolm_utils) and *plugins* (e.g. Saluki) in separate repositories.

- This repository provides the framework and a small plugin registry (`biolm_utils/plugin_registry.py`).
- Plugins should live in separate repositories (like `rna_saluki_cnn`) and register a lightweight factory that returns a `biolm_utils.config.Config` instance.

We include a thin example plugin here under `examples/plugin_template` and `saluki.py` demonstrates a thin Saluki wrapper that registers the plugin with the registry and exposes a small `activate()` helper for tests and wrappers.

### Packaging / entry-point

The canonical, supported packaging in this repository is the `saluki_plugin` package
(recommended). The root-level top-level implementation files were consolidated into
the `saluki_plugin/` package to avoid duplication and make the install flow unambiguous.

You can package this repository (or the `saluki_plugin` subpackage) so it exposes a
`biolm_utils.plugins` entry-point. For local development you can install the
plugin in editable mode:

```bash
# editable install (pip)
# install the packaged plugin (preferred):
pip install -e ./saluki_plugin
```

After installation the framework can discover the plugin using the
`discover_entrypoint_plugins()` helper or by calling the framework's
integration helpers (e.g., a CLI that calls discovery during startup).

### Quickstart — simulate training with Saluki (programmatic)

After you install the plugin into your environment (editable install during dev), you can run a minimal smoke training run that exercises the framework and the plugin registration. This demonstrates an end-to-end invocation without heavy datasets or long runtimes.

0. Install the framework (local dev only)

If you're developing the plugin alongside a local checkout of the framework, prefer
to use Poetry for environment management and then install the framework into the
Poetry environment so imports resolve correctly.

Example (recommended, Poetry):

```bash
# in the framework checkout
cd /path/to/biolm_utils
poetry install

# then in the plugin repo, install the framework into the same environment
cd /path/to/rna_saluki_cnn
poetry install
poetry run python -m pip install -e /path/to/biolm_utils
poetry run python -m pip install -e ./saluki_plugin

Quick bootstrap (one-liner)

If you prefer a single command you can use the included Makefile. From the
plugin repo root run (adjust FRAMEWORK_PATH if your framework checkout lives
elsewhere):

```bash
make bootstrap FRAMEWORK_PATH=/path/to/biolm_utils
```

This will create the Poetry environment (if needed), install the framework and
the plugin into the Poetry venv in editable mode so you can iterate locally.
```

If you don't use Poetry, the previous `pip install -e` approach still works inside a venv.

1. Install the plugin in editable mode:

```bash
cd /path/to/rna_saluki_cnn
.venv/bin/python -m pip install -e ./saluki_plugin
```

2. Run a small Python script to train a tiny model via biolm_utils (see demo in `examples/quick_train_saluki.py`):

```bash
.venv/bin/python examples/quick_train_saluki.py
```

3. Run the CLI via the framework

After the framework and plugin are installed, you can run the framework CLI to start a small training run or explore modes. Two common options:

- Run the framework CLI script directly from the framework checkout (recommended while developing framework + plugin locally):

```bash
# in a separate shell, go to the framework repo
cd /path/to/biolm_utils
.venv/bin/python biolm.py fine-tune # or other mode
```

- Or, run the framework module if the framework package is installed into your environment:

```bash
# run the framework CLI via the package
.venv/bin/python -m biolm_utils.biolm fine-tune
```

The framework attempts to discover installed plugins (entry-points group `biolm_utils.plugins`) at startup, so the Saluki plugin will be available to `biolm.py` after installation.

Note: when executed as a file, Python sets the import search path to the script's
directory which can hide packages in the repository root. The example script now
adds the repo root to sys.path automatically so it should work when invoked as
above. If you still hit import issues, run the script from the project root and
ensure the environment is using the project venv (or set PYTHONPATH=.).

The demo will perform a tiny training epoch (smoke) and should finish quickly. If everything succeeds, your plugin is correctly discoverable and compatible with the framework.

## Developer guide — single-env workflow (copy/paste)

We use a single shared Python environment for framework + plugin development — this is the simplest, most deterministic workflow and what we recommend for contributors.

1) Create the framework Poetry venv and install its dependencies:

```bash
cd /path/to/biolm_utils
poetry env use $(which python)   # optional: pick which system Python will back the venv
poetry install                  # creates the Poetry venv and installs framework deps
```

2) Install BOTH framework and plugin into the same environment (editable installs):

```bash
cd /path/to/rna_saluki_cnn
# find the same venv python that belongs to the framework venv
VENV_PYTHON=$(poetry --directory /path/to/biolm_utils env info -p)/bin/python
$VENV_PYTHON -m pip install -e /path/to/biolm_utils
$VENV_PYTHON -m pip install -e ./saluki_plugin
```

3) Run the quick demo (smoke test) using the same environment:

```bash
poetry run python examples/quick_train_saluki.py
```

Run tests & simulate CI locally
-------------------------------
To run the tests locally inside the plugin's Poetry venv and simulate the CI smoke flow:

```bash
# run basic unit tests
poetry run pytest -q tests/test_saluki_plugin.py

# run integration test that performs an editable install of the packaged plugin
poetry run pytest -q tests/test_plugin_install_integration.py

# convenience helper that creates the venv and installs framework + plugin into it
make bootstrap FRAMEWORK_PATH=/path/to/biolm_utils
poetry run python examples/quick_train_saluki.py
```

Notes
- The canonical packaging entry-point is `saluki_plugin/` (install with `pip install -e ./saluki_plugin`).
- Keep a single active venv for both framework and plugin development to avoid import/discovery confusion.


## Output layout

Experiments default to the `outputpath` in `params.py`. Typical layout:

```
my_experiment/
  tokenizer.json
  pre-train/
  fine-tune/<fold-id>/pytorch_model.bin
```

### Modes & examples

Main CLI: `biolm.py` (modes: `tokenize`, `pre-train`, `fine-tune`, `interpret`, `predict`).

Examples:

```bash
# tokenize
python biolm.py tokenize --configfile config.yaml

# pre-train
python biolm.py pre-train --filepath data.txt --outputpath out/

# fine-tune
python biolm.py fine-tune --filepath train.tsv --outputpath out/

# interpret / predict
python biolm.py interpret --inference.pretrainedmodel out/fine-tune/0
```

#### Notes

- `splitpos=None` → 90/10 train/val (no test). If you provide split ids, the code will run cross-validation over splits.
- `specifiersep` (one-hot only) allows per-token float channels (e.g. `A#2.5`).
- `vocabsize`: The maximal size of the vocabulary at the end of the tokenization process.
- `minfreq`: The minimum frequency that a token should appear in the training file before it is recorded as vocabulary item.
- `atomicreplacements`: This is a dictionary with tokens that should be treated as atomic tokens during the byte pair encoding process. You have to specify both: The initial token and the character that it is to be mapped to. 
- `encoding`: The encoding to apply: character-wise (`atomic`) or BPE (`bpe`).
- `maxtokenlength`: The BPE tokenizer can come up with pretty long tokens. This number caps the length at a maximal length.
- `lefttailing`: If true, sequences are cropped from the left (keeps right-side context).

### Pre-training (language models only) and fine-tuning a model 

For pre-training an language model via Masked Language Modelling you will use the `pre-train` mode. For fine-tuning a model, the `fine-tune` mode is required. In your `config.yaml` you need to at least specify the parameters under `training`:

```yaml
training:
  general:
    batchsize: 8
    gradacc: 4
    blocksize: 512
    nepochs: 10
    patience: 3
    resume: False # for resuming training
  fine-tuning:
    fromscratch: False # if we want to fine-tune without a pre-trained model (language models only)
    scaling: log # [log, minmax, standard]
    weightedregression: False
```

The attributes under `training: general` should be mostly self-explanatory: `blocksize` referes to the sequence length and might lead to errors when chosen bigger than `512` (for XLNET). For Saluki, we were able to set this maximum sequence length to `12288`. Sequences will then be truncated by the tokenizer or will be tokenized, re-centered and cropped when using the option `cdscentered` (see down below).

We also have to clarify data pre-processing and environment options:

```bash
data pre-processing:
  centertoken: False # either False or a token/character on which the sequence will be centered
environment:
  detected_ngpus: (auto-detected)  # Auto-detected; powers of two only (1,2,4,...)

BREAKING CHANGE: explicit GPU counts removed
------------------------------------------------
Note: The legacy `ngpus` option in `settings.environment` and `debugging.ngpus` has been removed. GPU counts are now auto-detected and exposed at `debugging.detected_ngpus` in the final `BioLMConfig` returned by `load_config()`.
 - Do not set `settings.environment.ngpus` or `debugging.ngpus` in your config YAMLs; they raise a ValueError.
 - Programmatic access: use `from biolm_utils.params import get_detected_ngpus` and call `get_detected_ngpus(args)`.
 - Example: `detected = get_detected_ngpus(args)`.
```

The `data processing` attributes refer to specific pre-processing options that are in detail explained by the command line help.

### Programmatic orchestration (train/dev/test runs with cross-validation)

If you want to orchestrate runs from other Python code (for example, to integrate
the library into a higher-level workflow or test harness) prefer the explicit
helpers introduced in the refactor: `make_run_fn`, `CrossValidator` and
`Paths`. These are easier to unit-test and avoid mutating global state.

Example (high-level):

```py
from biolm_utils.config import get_config
from biolm_utils.params import load_config
from biolm_utils.train_tokenizer import tokenize
from biolm_utils.train_utils import get_tokenizer, get_dataset
from biolm_utils.runner import make_run_fn
from biolm_utils.cross_validation import CrossValidator
from biolm_utils.paths import Paths

# Load your config / args (same objects used by the CLI)
config = get_config()
# load_config returns a BioLMConfig dataclass instance
args = load_config()

# Prepare tokenizer / datasets as usual
tokenizer = get_tokenizer(args, /* TOKENIZERFILE */, config.TOKENIZER_CLS, config.PRETRAINING_REQUIRED)
tokenizer_for_trainer = tokenizer
full_dataset = get_dataset(args, tokenizer, config.ADD_SPECIAL_TOKENS, /* DATASETFILE */, config.DATASET_CLS)

# Build the per-run callable (identical signature as legacy nested `run`):
run_once = make_run_fn(args, config, tokenizer, tokenizer_for_trainer, full_dataset)

# Create immutable per-run paths (these values come from biolm_utils.entry in the CLI)
base_paths = Paths(
  model_load_path=/* MODELLOADPATH */,
  model_save_path=/* MODELSAVEPATH */,
  output_path=/* OUTPUTPATH */,
  report_file=/* REPORTFILE */,
  rank_file=/* RANKFILE */,
)

# Instantiate CrossValidator and run the selected mode: fine-tune, predict, interpret, pre-train
cv = CrossValidator(params=args, dataset=full_dataset, run_once_fn=run_once, base_paths=base_paths)
result = cv.execute()

# `result` contains per-mode semantics (list of fold results for cross-validation, or a single value for predict)
```

## Configuration loader — programmatic usage and CLI behaviour

We simplified the configuration loader to be clearer and easier to test. Key
points you should be aware of:

- load_config now returns a structured `BioLMConfig` dataclass (no more implicit
  flattened argparse.Namespace). Use `cfg.data_source.filepath`,
  `cfg.training.batchsize`, `cfg.debugging.detected_ngpus`, etc.
- When calling programmatically prefer the explicit API: pass Hydra-style
  overrides as a list of strings (`key=value`). We purposely stopped auto-parsing
  sys.argv for programmatic calls — that behaviour was fragile and confusing.

Examples:

Programmatic:

```py
from biolm_utils.params import load_config

# Explicit list of overrides: 'key=value' strings
cfg = load_config(["mode=tokenize", "debugging.accelerator=cpu"])
print(cfg.mode)  # -> 'tokenize'
```

Via CLI (Hydra):

```bash
# Use Hydra-style overrides from the shell; Hydra CLI still works as before
python biolm.py mode=tokenize debugging.accelerator=cpu
```

Notes:

- Old behaviour where `load_config()` attempted to parse `sys.argv` and
  convert `--flag value` style arguments to hierarchical keys (e.g. `--filepath`
  -> `data_source.filepath`) has been removed. If you relied on that behaviour,
  update invocations to call `load_config` with explicit overrides or call the
  CLI directly (Hydra handles CLI args).
- Config validation and runtime GPU autodetection now live on the
  `BioLMConfig` dataclass via `cfg.validate()` and `cfg.autodetect_gpus()` and
  are run automatically when using `load_config`.

Notes & migration
- `run_once` keeps the original signature used by the old decorator: run(train, val, test, model_load, model_save, report, rank)
- The old `@parametrized_decorator` wrapper is still available for backward compatibility but is deprecated — prefer the `CrossValidator` + `make_run_fn` flow above.

### Cross-validation behaviour and pitfalls

Cross-validation configuration can be a little subtle — here are the rules and gotchas so you get deterministic, predictable behavior.

- `data_source.crossvalidation` accepts three kinds of values:
  - `null` / `0` / `False` (default) — no cross-validation. The code will either use `splitpos` + `devsplits` (deterministic splits) when provided, or a single random split when `splitratio` is specified.
  - `true` — *use predefined splits*. This requires `splitpos` to be set and `devsplits` (a list of split ids — and optionally `testsplits`) to be provided in your config or dataset file. This runs one pass per entry in `devsplits` (and `testsplits` if set) deterministically.
  - integer >= 2 — *random k-fold cross-validation* (k-fold). This performs k independent shuffled runs and requires `splitratio` (e.g., `[80,10,10]` or `[80,20]`) to determine train/val/(test) percentages. Note: `crossvalidation=1` is not allowed because it is ambiguous.

Pitfalls to avoid:
- `crossvalidation=true` without `splitpos` is ambiguous and will now raise an error — either provide `splitpos` (and `devsplits`) or set `crossvalidation` to a positive integer >= 2 and a `splitratio`.
- `crossvalidation` as an integer while `splitpos` is present is conflicting — numeric crossvalidation implies random splits and therefore conflicts with predefined split positions; prefer `crossvalidation=true` for predefined splits.
- `splitpos` set without `devsplits` is invalid — you must provide `devsplits` (and optionally `testsplits`) to define which splits are used for validation/testing.

Example YAML snippets:

1) Predefined splits (one deterministic CV run per entry of devsplits):

```yaml
data_source:
  splitpos: 3
  devsplits: [[1], [2]]  # list-of-lists: each tuple defines dev/test groupings
  testsplits: [[3], [4]] # optional
  crossvalidation: true
```

2) Random 5-fold cross-validation with 80/10/10 train/val/test:

```yaml
data_source:
  crossvalidation: 5
  splitratio: [80, 10, 10]
```

3) No CV (single run): deterministic with splits or a single random split

```yaml
data_source:
  crossvalidation: 0
  splitpos: 1
  devsplits: [2]
```

The library also validates these combinations early — invalid or ambiguous settings will raise a helpful error explaining the expected fix.

Automatic migration helper

To help migrate older configs that may use ambiguous forms, we've added a small helper in `biolm_utils.cfg_migration`:

- `analyze_crossvalidation(params)` — returns human-readable notes about ambiguous or problematic settings.
- `migrate_crossvalidation(params, auto_apply=False)` — returns a copy of `params` and recommended fixes; with `auto_apply=True` it will apply safe conversions (e.g. `0 -> False`, `True + splitratio -> convert to default k-fold`).

Usage example:

```py
from biolm_utils.cfg_migration import analyze_crossvalidation, migrate_crossvalidation

# analyze
notes = analyze_crossvalidation(args)
for n in notes:
  print("TODO:", n)

# apply safe migrations
new_args, applied_notes = migrate_crossvalidation(args, auto_apply=True)
```


Under `environment`, you can decide if you want to train on GPU or CPU and on how many GPUs you want to train. GPU count is auto-detected and restricted to powers-of-two values (1, 2, 4, 8...).

### Extract LOO-scores for a model

To calculate importance scores for indidvidual input tokens, we can use the mode `interpret`. The script will then run over the test splits and extracts leave-one-out (LOO) scores. The LOO scores are estimated by leaving a certain token blank (or delete comepletely, see options below), run the model with this "defective" sequence and compare the results to the prediction of the model for the original sequence. Positive scores denote, that leaving the input out leads to higher prediction, v.v. negative score means, leaving the input out leads to lower predictions. 

```yaml
looscores:
  handletokens: remove # remove, mask, replace
  replacementdict: None # dict of atomic tokens that should be replaced against each other if `--handletokens` is set to `replace`."
```

The scripts will then extract LOO scores for all splits of the fine-tuning data and saves them as `.csv` under the corresponding fine-tuning path as `loo_scores_{handle_tokens}.csv`.

### Inference:

Inference means sending a fine-tuned model on unseen data and let it make predictions. For this, run the main script with in the `predict` mode. The configfile mirrors only a fraction of the attributes compared to the complete pipeline.

### Resuming a model

There are two use cases to resume a model using the `--resume` argument:
1) `--resume` (without parameters) triggers the huggingface internal `resume_from_checkpoint` option which will only _continue_
a training that has been interrupted. For example, a planned training that was to run for 50 epochs and was interrupted  at epoch
23 can be resumed from the best checkpoint to be run from epoch 23 to planned epoch 50.
2) `--resume X` will trigger further pre-training a model from its best checkpoint for additional `X` epochs.


## Customization

This framwework on it's own does not provide full functionality. It is meant to be employed with plugins that implement the following classes and methods:
- A custom model class that inherits from 🤗 [PreTrainedModel](https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/model#transformers.PreTrainedModel) and provides a static `getconfig()` method.
- A custom dataset class that inherits from [RNABaseDataset](./biolm_utils/rna_datasets.py) and provides the `__getitem__()` method.
- A main script that imports the `run()` method from [biolm.py](./biolm_utils/biolm.py) and defines a custom `Config` object from [config.py](./biolm_utils/config.py) via `setconfig()`.

## License
