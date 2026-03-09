# Saluki Plugin Test Overview

This document explains what each test file in [tests](tests) validates.

## 1) [tests/test_saluki_plugin.py](tests/test_saluki_plugin.py)

Purpose: **entry-point smoke checks**.

What it tests:

- `saluki` entry point exists in Python metadata (`biolm.plugins` group).
- The entry-point target can be imported and called without crashing.

Why it matters:

- Confirms plugin installation/packaging wiring is correct.
- Catches broken `pyproject.toml` entry-point declarations quickly.

---

## 2) [tests/test_saluki_plugin_config.py](tests/test_saluki_plugin_config.py)

Purpose: **plugin contract + configuration correctness**.

What it tests:

- Plugin can be loaded from entry points into `PluginManager`.
- `pretraining_required == False` (Saluki is CNN-only, no pretrain stage).
- `model_cls_for_pretraining is None`.
- Required attributes exist (`model_cls_for_finetuning`, `dataset_cls`, tokenizer/data collator fields).
- Finetuning model class is callable.

Why it matters:

- Ensures Saluki obeys BioLM plugin contract semantics.
- Prevents regressions where config fields are renamed/removed.

---

## 3) [tests/test_saluki_invariants.py](tests/test_saluki_invariants.py)

Purpose: **data pipeline invariants specific to Saluki**.

What it tests:

- Dataset enforces Saluki blocksize invariant (`SALUKI_BLOCKSIZE`, 12288) even if config omits blocksize.
- Non-atomic tokenization is rejected (`tokenization.encoding` must be `atomic`).

Why it matters:

- Guards critical model assumptions.
- Fails early with a clear error if config is incompatible with Saluki architecture.

---

## 4) [tests/test_saluki_full_pipeline.py](tests/test_saluki_full_pipeline.py)

Purpose: **end-to-end integration test** of Saluki workflow.

What it tests:

- Full run: tokenize -> fine-tune -> evaluate (regression).
- Produces expected artifacts (tokenizer file, fine-tune outputs).
- Evaluation output includes `eval_spearman rho`.

Why it matters:

- Verifies all moving parts work together (CLI/Hydra/config/dataset/model/training/eval).
- Most realistic health check, but also slowest test.

---

## Practical notes

- The full pipeline test is intentionally heavier and can take a few minutes.
- If entry-point tests fail first, check plugin installation and environment activation before debugging model code.
- Recommended triage order:
  1. `test_saluki_plugin.py`
  2. `test_saluki_plugin_config.py`
  3. `test_saluki_invariants.py`
  4. `test_saluki_full_pipeline.py`
