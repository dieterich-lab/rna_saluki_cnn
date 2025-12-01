"""Tests ensuring the Saluki plugin can be loaded via entry points."""

import importlib.metadata


def test_saluki_plugin_entry_point_exists():
    """Test that the saluki plugin entry point is registered."""
    eps = importlib.metadata.entry_points(group="biolm.plugins")
    plugin_names = [ep.name for ep in eps]
    assert "saluki" in plugin_names, f"saluki not in {plugin_names}"


def test_saluki_plugin_can_load():
    """Test that the saluki plugin can be loaded."""
    eps = importlib.metadata.entry_points(group="biolm.plugins")
    for ep in eps:
        if ep.name == "saluki":
            plugin_func = ep.load()
            # Call the function to load the plugin
            plugin_func()
            break
    else:
        raise AssertionError("saluki entry point not found")
