Plugin template and registration guide
====================================

This project supports a lightweight plugin mechanism for model/dataset/tokenizer
plugins. The recommended pattern for a plugin (for educational projects) is:

1. Add a factory function `get_<plugin>_config()` that returns a `biolm_utils.config.Config`.
2. Register that factory with `biolm_utils.plugin_registry.register_plugin("name", factory)`.
3. Optionally apply it for immediate use in a script with `apply_plugin("name")`.

Example (see `biolm_utils/examples/plugin_template.py`):

```python
from biolm_utils.plugin_registry import register_plugin, apply_plugin
from your_plugin import get_your_config

register_plugin("your_plugin", get_your_config)
apply_plugin("your_plugin")  # now biolm_utils.get_config() returns your plugin config
```

This simple API makes it easy for users to create discrete plugins without
modifying the core codebase, while keeping the system testable and explicit.
