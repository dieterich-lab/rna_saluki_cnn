# Plugin template (Saluki-style)

This small example shows the minimal layout expected for a plugin that integrates
with the `biolm_utils` framework.

Structure:
- `plugin_skeleton.py` — factory function + registration using
  `biolm_utils.plugin_registry.register_plugin("my_plugin", factory)`.
- `dataset.py` / `model.py` — plugin-local dataset & model classes (lightweight)
- `tests/test_plugin_template.py` — a tiny test to ensure registration works.

The template demonstrates a thin, importable factory that returns a
`biolm_utils.config.Config` dataclass, avoiding heavy side-effects on import.
