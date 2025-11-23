"""Small plugin registry for model/dataset/tokenizer plugins.

This provides a light-weight programmatic API plugins can use to register
themselves with the `biolm_utils` host project. Tests and the top-level
`saluki.py` can register a plugin and then apply it to set the active
configuration via `set_config`.

The registry intentionally stays minimal to keep plugin examples simple for
educational use.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from .config import Config, set_config

_REGISTRY: Dict[str, Callable[[], Config]] = {}
# Stack of applied plugin (name, config) to allow reverting to previous plugin
_APPLIED_STACK: list[tuple[Optional[str], Optional[Config]]] = []
_ACTIVE_PLUGIN: Optional[str] = None


def register_plugin(name: str, factory: Callable[[], Config]) -> None:
    """Register a plugin factory under a name.

    The factory should return a `Config` object when called.
    """
    if name in _REGISTRY:
        raise RuntimeError(f"Plugin '{name}' already registered")
    _REGISTRY[name] = factory


def list_plugins() -> list[str]:
    return list(_REGISTRY.keys())


def get_plugin_factory(name: str) -> Optional[Callable[[], Config]]:
    return _REGISTRY.get(name)


def apply_plugin(name: str) -> None:
    """Call the factory for plugin `name` and set the returned Config as active.

    Raises RuntimeError if the plugin is not registered.
    """
    factory = get_plugin_factory(name)
    if factory is None:
        raise RuntimeError(f"Plugin '{name}' not found")
    global _ACTIVE_PLUGIN
    # capture current config (if any) so we can restore later
    try:
        from .config import get_config

        current = get_config()
    except Exception:
        current = None

    _APPLIED_STACK.append((_ACTIVE_PLUGIN, current))
    config = factory()
    set_config(config)
    _ACTIVE_PLUGIN = name


def get_current_plugin() -> Optional[str]:
    return _ACTIVE_PLUGIN


def get_applied_stack() -> list[tuple[Optional[str], Optional[Config]]]:
    return list(_APPLIED_STACK)


def unregister_plugin(name: str) -> None:
    """Remove a plugin from the registry.

    Raises KeyError if the plugin is not found. Does not alter applied stack.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Plugin '{name}' not registered")
    del _REGISTRY[name]


def restore_previous_plugin() -> None:
    """Restore the previously active plugin (pop the applied stack).

    If no previous plugin exists this will simply clear the active plugin and
    leave the configuration unchanged if there is nothing to restore.
    """
    global _ACTIVE_PLUGIN
    if not _APPLIED_STACK:
        _ACTIVE_PLUGIN = None
        return

    prev_name, prev_config = _APPLIED_STACK.pop()
    if prev_config is not None:
        set_config(prev_config)
    _ACTIVE_PLUGIN = prev_name
