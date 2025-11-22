# `biolm_utils`: A framework to run bioinformatical Language Models.

This projects implements pre-training and fine-tuning of neural models for regressing half lives of RNA and protein sequences. In addition, it supports the extraction of leave-one-out (LOO) scores for fine-tuned models to analyse importance scores of individual inputs.

In detail, the following steps are implemented:

- Tokenization of RNA/Protein sequences via 
  - Byte Pair encoding
  - atomic one-hot encoding
- Pre-train a  language model via Masked Language Modelling.
- Fine-tune any model for regressing half lives.
- Calculation of leave-one-out scores for you fine-tuned model.

## Installation


```bash
git clone https://github.com/dieterich-lab/biolm_utils.git
cd biolm_utils

# Recommended: use Poetry for reproducible environments
poetry install
```
We now recommend using Poetry for reproducible environments. If you haven't installed Poetry, follow the instructions at https://python-poetry.org.

Quick setup using Poetry (recommended):

```bash
git clone https://github.com/dieterich-lab/biolm_utils.git
cd biolm_utils
# (optional) use a specific Python interpreter
poetry env use $(which python)
poetry install
poetry run pytest -q
```

Notes:
- The project ships `pyproject.toml`. Adding a `poetry.lock` into the repo will
  make installs fully reproducible across machines and CI. CI now supports a
  Poetry-based workflow (`.github/workflows/poetry-tests.yml`).

If you'd like to use MLflow locally for experiments or demos, install the optional
MLflow group via Poetry (supported in modern Poetry versions) with:

```bash
poetry install --with mlflow
```

If your Poetry version does not support groups/extras you can still use pip to
install MLflow into the project's virtual environment (generic example):

```bash
# install into the project's venv (run from the project root or your active venv)
python -m pip install mlflow
```

Migration note (Pipfile -> Poetry)
----------------------------------

This project previously used a Pipfile / Pipenv workflow. The recommended path
forward is Poetry (pyproject.toml + poetry.lock). To migrate existing environments
or CI from Pipenv, initialize a Poetry project in the repo root and generate a
lockfile:

```bash
poetry init --no-interaction
poetry lock
```

Once you've validated that Poetry works for your workflow you can remove the
legacy `Pipfile` and `Pipfile.lock` files.

## File structure

```bash
├── biolm_utils
│   ├── biolm.py # Main script for tokenizing, training, testing and predicting and loo sores.
│   ├── config.py # Config class that needs to be initalized by plugings .
│   ├── cross_validation.py # Contains the newer `CrossValidator` orchestration. A compatibility shim `parametrized_decorator` remains but is deprecated (see `DEPRECATE_PARAMETRIZED_DECORATOR.md`).
│   ├── entry.py # After params.py, this is the main entry point of the program, fixing paths and global variables .
│   ├── __init__.py
│   ├── interpret.py # Script controlling the loo score calculation.
│   ├── loo_utils.py # Contains a custom evaluator to extract LOO scores for regression tasks.
│   ├── params.py # Argparser.
│   ├── rna_datasets.py # Dataset class handling tokenized and vectorized sequences.
│   ├── trainer.py # Custom trainer classes that can fine-tune a model for regression tasks.
│   ├── train_tokenizer.py # Script controlling the tokenzation processing.
│   └── train_utils.py # Contains various helper functions, e.g. to get load models/tokenizer or create reports..
├── pyproject.toml
└── README.md
```

## Documentation & teaching guides

- The project contains a short pedagogical guide covering the framework internals and how to write a compatible plugin (example: Saluki):
  - DOCS/framework_and_saluki_plugin_guide.md


## Pathing

The software will save all experiment data in the `outputpath` given in [params.py](biolm_utils/params.py) (or fall back to the file path stem of the input file givein in `filepath` if not given). This directory will be created if not existant. There, we will save the dataset (tokenized samples from the given filepath), the tokenizer and the models. I.e. considering we use cross valdiation via splits and after having pre-trained (language models only) and fine-tuned a model, the directory will look as follows:

```bash
├── my_experiment
│   ├── fine-tune
│   │   ├── 0
│   │   │   └── pytorch_model.bin
│   │   ├── 1
│   │   │   └── pytorch_model.bin
│   │   ├── 2
│   │   │   └── pytorch_model.bin
│   │   └── dataset.json
│   ├── pre-train
│   │   ├── dataset.json
│   │   └── pytorch_model.bin
│   └── tokenizer.json
```

## Usage

### General

The main script is [biolm.py](biolm_utils/biolm.py). It contains a `run()` function that can be imported into your custom project. It will access the given parameters from the parameters in [`params.py`](biolm_utils/params.py) and additionally from a custom `Config` object located in [config.py](/biolm_utils/config.py) that can be set via `set_config()`.

To get a verbose exlplanation on all the possible parameters you can run the following:


```bash
python biolm.py -h 
```

All options besides the training `mode` are optional and are mostly populated with sensible default parameters. The `mode` can be one of the following:

- `tokenize` 
- `pre-train`
- `fine-tune`
- `interpret`
- `predict`

As an example, you can run training with command line parameters

```bash
python biolm.py pre-train --filepath "xxx" --outputpath "xxx" --...

```
or start tokenization with a config file

```bash
python biolm.py tokenize --configfile {config.yaml}
```

The parameters in the config file will then be parsed by the argparser in [params.py](/biolm_utils/params.py) to rule out any conflicts. Parameters parsed from the command line have priority over those from the config file.

### Configuring the data

We designed options to give varying data sources for either tokenzation/and pre-training (we expect that the data for training the tokenizer will be the same as for the pre-training process) and for the fine-tuning step. You also have to let the scripts know where exactly to find information about labels, sequences and splits in your data file. The two according sections in the config file are listed below. Attributes should be self-explanatory by their comments or explained by the command line parser. 

```yaml
#
# Description of the datasource used for 
# - training the tokenizer 
# - pre-training (for LM)
#
tokenizing and pre-training data source:
  filepath: "tokenizing_and_pre-training_data_file.txt"
  stripheader: False # if the custom data file has a header that has to be stripped
  columnsep: "\t" # could be "," "|", "\t" ...
  tokensep: ","
  specifiersep: None
  idpos: 1 # position of the identifier of the column 
  seqpos: 1 # position of the sequence column 
  pretrainedmodel: None # if the tokenizer for pre-training diverts from the chosen data.

#
# Description of the fine-tuning source
#
fine-tuning data source:
  filepath: "fine-tuning_data_file.txt"
  stripheader: False # if the custom data file has a header that has to be stripped
  columnsep: "\t" # could be "," "|", "\t" ...
  tokensep: ","
  specifiersep: None
  idpos: 1 # position of the identifier of the column 
  seqpos: 1 # position of the sequence column 
  labelpos: 1 # position of the label column 
  weightpos: None # position of the column containing quality labels 
  splitpos: 1 # position of the split identifier for cross validaton
  pretrainedmodel: None # if the pre-trained model diverts from the chosen data.
  ```

An example prototypical dataset file would look like this (without header)

```csv
0	ENST00000488147	ENSG00000227232	653635	WASH7P	unprocessed_pseudogene	0.204213162843933	3.39423360819142	0.121582579281952	0.374739086478062	a,t,g,g,g,a,g,c,c,g,t,g,t,g,c,a,c,g,t,c,g,g,g,a,g,c,t,c,g,g,a,g,t,g,a,g,c,gej,c,a,c,c,a,t,g,a,c,t,c,c,t,g,t,g,a,g,g,a,t,g,c,a,g,c,a,c,t,c,c,c,t,g,g,c,a,g,g,t,c,a,g,a,c,c,t,a,t,g,c,c,g,t,g,c,c,c,t,t,c,a,t,c,c,a,g,c,c,a,g,a,c,c,t,g,c,g,g,c,g,a,g,a,g,g,a,g,g,c,c,g,t,c,c,a,g,c,a,g,a,t,g,g,c,g,g,a,t,g,c,c,c,t,g,c,a,g,t,a,c,c,t,g,c,a,g,a,a,g,g,t,c,t,c,t,g,g,a,g,a,c,a,t,c,t,t,c,a,g,c,a,g,gej,t,a,g,a,g,c,a,g,a,g,c,c,g,g,a,g,c,c,a,g,g,t,g,c,a,g,g,c,c,a,t,t,g,g,a,g,a,g,a,a,g,g,t,c,t,c,c,t,t,g,g,c,c,c,a,g,g,c,c,a,a,g,a,t,t,g,a,g,a,a,g,a,t,c,a,a,g,g,g,c,a,g,c,a,a,g,a,a,g,g,c,c,a,t,c,a,a,g,gej,t,g,t,t,c,t,c,c,a,g,t,g,c,c,a,a,g,t,a,c,c,c,t,g,c,t,c,c,a,g,g,g,c,g,c,c,t,g,c,a,g,g,a,a,t,a,t,g,g,c,t,c,c,a,t,c,t,t,c,a,c,g,g,g,c,g,c,c,c,a,g,g,a,c,c,c,t,g,g,c,c,t,g,c,a,g,a,g,a,c,g,c,c,c,c,c,g,c,c,a,c,a,g,g,a,t,c,c,a,g,a,g,c,a,a,g,c,a,c,c,g,c,c,c,c,c,t,g,g,a,c,g,a,g,c,g,g,g,c,c,c,t,g,c,a,g,gej,a,g,a,a,g,c,t,g,a,a,g,g,a,c,t,t,t,c,c,t,g,t,g,t,g,c,g,t,g,a,g,c,a,c,c,a,a,g,c,c,g,g,a,g,c,c,c,g,a,g,g,a,c,g,a,t,g,c,a,g,a,a,g,a,g,g,g,a,c,t,t,g,g,g,g,g,t,c,t,t,c,c,c,a,g,c,a,a,c,a,t,c,a,g,c,t,c,t,g,t,c,a,g,c,t,c,c,t,t,g,c,t,g,c,t,c,t,t,c,a,a,c,a,c,c,a,c,c,g,a,g,a,a,c,c,t,gej,t,a,g,a,a,g,a,a,g,t,a,t,g,t,c,t,t,c,c,t,g,g,a,c,c,c,c,c,t,g,g,c,t,g,g,t,g,c,t,g,t,a,a,c,a,a,a,g,a,c,c,c,a,t,g,t,g,a,t,g,c,t,g,g,g,g,g,c,a,g,a,g,a,c,a,g,a,g,g,a,g,a,a,g,c,t,g,t,t,t,g,a,t,g,c,c,c,c,c,t,t,g,t,c,c,a,t,c,a,g,c,a,a,g,a,g,a,g,a,g,c,a,g,c,t,g,g,a,a,c,a,g,c,a,g,gej,t,c,c,c,a,g,a,g,a,a,c,t,a,c,t,t,c,t,a,t,g,t,g,c,c,a,g,a,c,c,t,g,g,g,c,c,a,g,g,t,g,c,c,t,g,a,g,a,t,t,g,a,t,g,t,t,c,c,a,t,c,c,t,a,c,c,t,g,c,c,t,g,a,c,c,t,g,c,c,c,g,g,c,a,t,t,g,c,c,a,a,c,g,a,c,c,t,c,a,t,g,t,a,c,a,t,t,g,c,c,g,a,c,c,t,g,g,g,c,c,c,c,g,g,c,a,t,t,g,c,c,c,c,c,t,c,t,g,c,c,c,c,t,g,g,c,a,c,c,a,t,t,c,c,a,g,a,a,c,t,g,c,c,c,a,c,c,t,t,c,c,a,c,a,c,t,g,a,g,g,t,a,g,c,c,g,a,g,c,c,t,c,t,c,a,a,g,aej,c,c,t,a,c,a,a,g,a,t,g,g,g,g,t,a,c,t,a,a,c,a,c,c,a,c,c,c,c,c,a,c,c,g,c,c,c,c,c,a,c,c,a,c,c,a,c,c,c,c,c,a,g,c,t,c,c,t,g,a,g,g,t,g,c,t,g,g,c,c,a,g,t,g,c,a,c,c,c,c,c,a,c,t,c,c,c,a,c,c,c,t,c,a,a,c,c,g,c,g,g,c,c,c,c,t,g,t,a,g,g,c,c,a,a,g,g,c,g,c,c,a,g,g,c,a,g,g,a,c,g,a,c,a,g,c,a,g,c,a,g,c,a,g,c,g,c,g,t,c,t,c,c,t,t,c,a,g,tej,c,c,a,g,g,g,a,g,c,t,c,c,c,a,g,g,g,a,a,g,t,g,g,t,t,g,a,c,c,c,c,t,c,c,g,g,t,g,g,c,t,g,g,c,c,a,c,t,c,t,g,c,t,a,g,a,g,t,c,c,a,t,c,c,g,c,c,a,a,g,c,t,g,g,g,g,g,c,a,t,c,g,g,c,a,a,g,g,c,c,a,a,g,c,t,g,c,g,c,a,g,c,a,t,g,a,a,g,g,a,g,c,g,a,a,a,g,c,t,g,g,a,g,a,a,g,c,a,g,c,a,g,c,a,g,a,a,g,g,a,g,c,a,g,g,a,g,c,a,a,g,tej,g,a,g,a,g,c,c,a,c,g,a,g,c,c,a,a,g,g,t,g,g,g,c,a,c,t,t,g,a,t,g,t,c,gej,c,t,c,c,a,t,g,g,g,g,g,g,a,c,g,g,c,t,c,c,a,c,c,c,a,g,c,c,t,g,c,g,c,c,a,c,t,g,t,g,t,t,c,t,t,a,a,g,a,g,g,c,t,t,c,c,a,g,a,g,a,a,a,a,c,g,g,c,a,c,a,c,c,a,a,t,c,a,a,t,a,a,a,g,a,a,c,t,g,a,g,c,a,g,a,a,a
```

There are certain specifics regarding the following entries:

- `splitpos`: If it is set to `None` fine-tuning is carried out on a 90/10 train/val split with no subsequent testing. If a splits position is given, we expect at least three different splits on which we do cross validation by:
  - setting each split as a dedicated test set
  - setting the following split as a dedicated validation set
  - and training on the rest of the splits.

- `specifiersep` (**one-hot encoding only**): If you want to decorate your atomic tokens with float numbers you can do so, by denoting a separator after which you append the float number(s) to the atomic token. For example, you could specify `specifiersep: #` for generating your samples as: `a#2.5, c, A, g#5.7, ...` or even with multiple modiefiers like `a#2.5#0.2, c, A, g#5.7, ...` . The decorating float numbers are then appended to new "channels" of the one-hot encoding. Regarding the last sample from above, this would result in a one-hot-encoding of (assuming a vocabulary of `[a, c, g, t, A, C, G, T]`):

```
a | 1  | 0 | 0 | 0 |
c | 0  | 1 | 0 | 0 |
g | 0  | 0 | 0 | 1 |
t | 0  | 0 | 0 | 0 |
A | 0  | 0 | 1 | 0 |
C | 0  | 0 | 0 | 0 |
G | 0  | 0 | 0 | 0 |
T | 0  | 0 | 0 | 0 |
  |2.5 | 0 | 0 |5.7|
  |0.2 | 0 | 0 |5.7|
```



### Training a tokenizer

To train a tokenizer, you'll be using the `tokenize` mode. The `encoding` parameter in the config file offers different encoding options. Under the section `tokenization` you'll find options to further customize the encoding process. 

```yaml

tokenization:
  samplesize: None # if your data is to big to learn a tokenizer, you can downsample it
  vocabsize: 20_000
  minfreq: 2
  atomicreplacements: None # dictionary of replacements, i.e. `{"a": "A", "bcd": "xyz"}
  encoding: atomic # [bpe, atomic]
  bpe:
    maxtokenlength: 10
  lefttailing: True
```

Where 

- `samplesize` offers the option to downsample your data. If you file has, for example, 10M lines, training a BPE tokenizer on all these might become very costly or computationally infeasible. You can instead give a smaplesize of `250_000` to make the process much faster.
- `vocabsize`: The maximal size of the vocabulary at the end of the tokenization process.
- `minfreq`: The minimum frequency that a token should appear in the training file before it is recorded as vocabulary item.
- `atomicreplacements`: This is a dictionary with tokens that should be treated as atomic tokens during the byte pair encoding process. You have to specify both: The initial token and the character that it is to be mapped to. 
- `encoding`: The actual encding to be apllied. Either characterwise (`atomic`) or using a word piece tokenizer for byte pair encoding (`bpe`).
- `maxtokenlength`: The BPE tokenizer can come up with pretty long tokens. This number caps the length at a maximal length.
- `lefttailing`: If true, the sequences will be cut from the left (begging from the right end).

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
Note: The legacy `ngpus` option in `settings.environment` and `debugging.ngpus` has been removed. GPU counts are now auto-detected and exposed at `debugging.detected_ngpus` in the final `Namespace` returned by `load_config()`.
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
from types import SimpleNamespace

from biolm_utils.config import get_config
from biolm_utils.params import load_config
from biolm_utils.train_tokenizer import tokenize
from biolm_utils.train_utils import get_tokenizer, get_dataset
from biolm_utils.runner import make_run_fn
from biolm_utils.cross_validation import CrossValidator
from biolm_utils.paths import Paths

# Load your config / args (same objects used by the CLI)
config = get_config()
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
