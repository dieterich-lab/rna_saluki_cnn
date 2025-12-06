# BioLM Utils — Framework + Saluki Plugin Guide

This document explains the structure of the biolm_utils framework and how a plugin (example: Saluki) integrates with it.

Audience: students, plugin authors, and maintainers who need an educational overview with runnable examples.

Contents
- Overview: concepts & responsibilities
- Core components (Config, runner, CrossValidator, Paths)
- Plugin architecture & registry API
- Saluki plugin walkthrough (example plugin implementation)
- Migration & testing guidance (including learning_rate canonicalization)
- Exercises for students

---

## 1 — Overview

biolm_utils is a lightweight orchestration framework for RNA modelling workflows (tokenize → fine-tune → predict → interpret), built to be clear and testable for teaching. The main ideas:

- Config: a single dataclass `Config` centralises model/dataset/tokenizer classes and runtime constants.
- Runner factory: `make_run_fn` returns a per-split function used by a CrossValidator to run training/eval consistently.
- CrossValidator: orchestrates k-fold experiments by invoking `run_fn` for each fold and collecting results.
- Plugin registry: register plugin factories that return `Config` objects and apply them at runtime.

This separation supports both unit testing and real experiments without global-hidden state.

---

## 2 — Core components

### 2.1 Config dataclass

`biolm_utils.config.Config` is the single trusted shape that plugins return.

Example (construct canonical snake_case):

```python
from biolm_utils.config import Config

conf = Config(
    model_cls_for_pretraining=None,
    model_cls_for_finetuning=MyModelClass,
    tokenizer_cls=MyTokenizer,
    learning_rate=1e-4,
    max_grad_norm=1.0,
    weight_decay=0.0,
    special_tokenizer_for_trainer_cls=None,
    datacollator_cls_for_pretraining=None,
    datacollator_cls_for_finetuning=DefaultDataCollator,
    add_special_tokens=False,
    config_cls=SomeTransformersConfig,
    pretraining_required=False,
    dataset_cls=MyDatasetClass,
)

from biolm_utils.config import set_config, get_config
set_config(conf)
current = get_config()
assert current is conf
```

Notes:
- `Config` is deliberately explicit: use named kwargs in new code so your code remains robust to positional ordering changes.

### 2.2 Runner factory and Paths

`runner.make_run_fn` returns a function that performs a single run for one cross-validation split. The `Paths` dataclass encapsulates experiment paths for saving models and outputs.

Example skeleton:

```python
from biolm_utils.runner import make_run_fn
from biolm_utils.paths import Paths

def my_plugin_run(paths: Paths, dataset):
    # implements training or evaluation, using classes from get_config()
    pass

run_fn = make_run_fn(my_plugin_run, some_shared_context)
# CrossValidator will later call run_fn(paths, split_index, indices)
```


### 2.3 CrossValidator

The `CrossValidator` class provides explicit, testable orchestration for k-fold experiments.

Usage example:

```python
from biolm_utils.cross_validation import CrossValidator

cv = CrossValidator(make_run_fn(…))
results = cv.run(dataset, k=5)
```

CrossValidator intentionally avoids decorators and global mutations in order to make unit testing and reuse simpler.

---

## 3 — Plugin architecture & registry API

biolm_utils supports a small plugin API so model/dataset/tokenizer implementations can be provided without modifying core code.

Key functions (brief):

- register_plugin(name: str, factory: Callable) — register a plugin factory.
- apply_plugin(name: str) — make a plugin the active config (calls set_config internally).
- unregister_plugin(name: str) — remove a plugin from registry.
- restore_previous_plugin() — restore prior applied plugin state.
- get_current_plugin(), list_plugins() — introspection helpers.

Example minimal plugin factory (recommended pattern uses named kwargs):

```python
# your_plugin.py
from transformers import BertConfig, PreTrainedTokenizerFast, DefaultDataCollator
from biolm_utils.config import Config

def get_your_config():
    return Config(
        model_cls_for_pretraining=None,
        model_cls_for_finetuning=MyModelClass,
        tokenizer_cls=PreTrainedTokenizerFast,
        learning_rate=1e-3,
        max_grad_norm=0.4,
        weight_decay=0.001,
        special_tokenizer_for_trainer_cls=None,
        datacollator_cls_for_pretraining=None,
        datacollator_cls_for_finetuning=DefaultDataCollator,
        add_special_tokens=False,
        config_cls=BertConfig,
        pretraining_required=False,
        dataset_cls=MyDatasetClass,
    )

# register the plugin
from biolm_utils.plugin_registry import register_plugin, apply_plugin
register_plugin('my_plugin', get_your_config)
# make it active
apply_plugin('my_plugin')

```

Important: plugin factories should be side-effect free and return a `Config` or a dataclass-like mapping to keep tests simple.

---

## 4 — Saluki: example plugin walkthrough

Saluki is an example plugin implemented in the separate `rna_saluki_cnn` repository (not embedded inside this project). The Saluki repo bundles `biolm_utils` as a local subtree / path dependency and demonstrates how to register and apply a plugin that integrates with this framework.
See the rna_saluki_cnn repository for a production plugin example and the top-level `saluki.py` wrapper.

Key points:

- Saluki provides a `get_saluki_config()` factory that constructs a `Config` using named kwargs.
- It registers the plugin with `register_plugin('saluki', get_saluki_config)` and commonly calls `apply_plugin('saluki')` in its wrapper script so the plugin becomes active for tests and CLI entry.
- Saluki then calls the framework's `main()` entrypoint (from `biolm_utils.biolm`) to run tokenization / fine-tuning / predict / interpret using the plugin's classes.

Example from `saluki.py` (simplified):

```python
from biolm_utils.plugin_registry import register_plugin, apply_plugin
from your_saluki_impl import get_saluki_config

register_plugin('saluki', get_saluki_config)
apply_plugin('saluki')

# call the general entrypoint — now get_config() resolves to the Saluki values
from biolm_utils.biolm import main
main()
```

Saluki also demonstrates a few plugin-specific invariants (tokenization encoding and fixed block size). Plugins can and should validate invariants early to surface clear errors.

---

## 5 — Migration & canonical naming

We recommend `learning_rate` (snake_case) as the canonical configuration key. A one-time breaking migration was performed in this repo to remove the backward compatibility shim. If you maintain plugins or external trees, follow this approach:

1. Keep a compatibility shim in your consumer until you can perform a repo-wide migration.
2. Write unit tests to verify both old and new names if you temporarily require compatibility.
3. Use an AST-driven script to do an organized rewrite of small cases (function-call keywords, config YAML keys). Keep backups before writing changes.

Example AST snippet (basic):

```python
import ast

legacy_key = '<legacy-key-name-to-replace>'
class RewriteKeywords(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call):
        for kw in node.keywords:
            # compare against a variable (set `legacy_key` before running)
            if kw.arg == legacy_key:
                kw.arg = 'learning_rate'
        return node

# apply on file contents and write back safely
```

We recommend human review for non-trivial code sites and a follow-up PR that removes the shim from the runtime once all known dependents are migrated.

---

## Optional MLflow integration

The framework supports an opt-in MLflow integration that can track runs, metadata, metrics and artifacts per cross-validation fold.

Key points:

- Disabled by default. Enable by setting `settings.mlflow.enabled=true` in Hydra configs.
- Runs are started automatically when the runner executes a per-fold run (CrossValidator). Each fold becomes a separate MLflow run.
- The integration will record plugin `Config` fields and selected top-level args (mode/task) as parameters and can optionally upload model_artifacts.

Enable and run example:

```yaml
settings:
    mlflow:
        enabled: true
        tracking_uri: file:///path/to/local/mlruns
        experiment_name: saluki-experiments
        log_artifacts: true
        run_name_template: "{mode}-fold-{path}"
```

Install the optional dependency and run (with Poetry extras or pip):

```bash
# poetry
poetry install --with mlflow

# pip
pip install mlflow
```

The integration is intentionally conservative and avoids hard-coupling (it lazy-imports MLflow and will present a clear error message if MLflow is enabled but not installed).

---

## 6 — Testing & developer workflow

Teach your students to:

1. Build small, isolated unit tests for each plugin factory (e.g., does it return a valid `Config`?).
2. Use the `make_run_fn` to create tiny integration tests that mock heavy dependencies and still exercise training logic.
3. Use the plugin registry in tests to register a short-lived plugin and apply it. Clean up after tests by unregistering.

Quick sample tests (pytest-style):

```python
def test_my_plugin_factory():
    cfg = get_your_config()
    assert isinstance(cfg, Config)
    assert cfg.learning_rate > 0

def test_register_and_apply(mocker):
    register_plugin('tmp', get_your_config)
    apply_plugin('tmp')
    assert get_config().model_cls_for_finetuning == MyModelClass
    unregister_plugin('tmp')
```

---

## 7 — Teacher exercises

1. Implement a tiny plugin that wraps a simple PyTorch linear model and dataset (no transformers) and integrate it with `make_run_fn` and `CrossValidator`.
2. Add a new Saluki variant that enforces a different tokenizer invariant; write tests asserting the plugin prevents invalid configs.
3. Perform the formal migration of any third-party plugin code that still uses legacy key names (for example, a pre-refactor name).

---

## 8 — Where to look in the codebase

- `biolm_utils/config.py` — main Config dataclass, set/get
- `biolm_utils/runner.py` — run factory
- `biolm_utils/cross_validation.py` — CrossValidator orchestration
- `biolm_utils/plugin_registry.py` — registration API
- `rna_saluki_cnn/saluki.py` — example plugin entrypoint (Saluki)

---

If you'd like, I can also:

- Add this guide to the project's main README with a short link.
- Produce an in-class lab worksheet that walks students through building a plugin and running tests.

Tell me whether you'd like the guide added to the README and/or expanded further into a small teacher-ready assignment (I can add tests and a sample plugin scaffold for hands-on exercises).
