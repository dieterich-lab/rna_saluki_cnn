# bioml_utils — utilities for bioinformatic language models

A compact toolkit for tokenizing, pre-training and fine-tuning language models on biological sequences (RNA/protein). It also supports interpretation with leave-one-out (LOO) scores.

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
