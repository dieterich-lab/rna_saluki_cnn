"""Plugin discovery helpers (mirror from framework) — kept in the Saluki repo
so tests can import the helper when the framework is included as a subtree.
"""

from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path
from typing import Optional

from .plugin_registry import register_plugin, get_plugin_factory


def discover_entrypoint_plugins(group: str = "biolm_utils.plugins") -> list[str]:
    registered = []
    eps = entry_points()
    try:
        group_eps = eps.select(group=group)
    except Exception:
        group_eps = [ep for ep in eps if getattr(ep, "group", None) == group]

    for ep in group_eps:
        name = ep.name
        if get_plugin_factory(name) is not None:
            continue
        try:
            factory = ep.load()
        except Exception:
            continue
        register_plugin(name, factory)
        registered.append(name)

    return registered


def discover_plugins_from_dir(plugins_dir: Optional[Path | str] = None) -> list[str]:
    registered = []
    path = Path(plugins_dir or "plugins")
    if not path.exists() or not path.is_dir():
        return registered

    for child in sorted(path.iterdir()):
        if child.name.startswith("."):
            continue

        module_name = None
        if child.is_file() and child.suffix == ".py":
            module_name = child.stem
        elif child.is_dir() and (child / "__init__.py").exists():
            module_name = child.name

        if module_name is None:
            continue

        try:
            mod = import_module(module_name)
        except Exception:
            continue

        if hasattr(mod, "register_plugin"):
            try:
                mod.register_plugin()
                registered.append(module_name)
            except Exception:
                continue
            continue

        factory = None
        factory_name = f"get_{module_name}_config"
        if hasattr(mod, factory_name):
            factory = getattr(mod, factory_name)
        elif hasattr(mod, "get_config"):
            factory = getattr(mod, "get_config")

        if callable(factory):
            register_plugin(module_name, factory)
            registered.append(module_name)

    return registered


def discover_all_plugins() -> list[str]:
    names = discover_entrypoint_plugins()
    names += [n for n in discover_plugins_from_dir() if n not in names]
    return names
