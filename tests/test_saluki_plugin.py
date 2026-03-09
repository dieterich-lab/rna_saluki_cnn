"""Tests ensuring the Saluki plugin can be loaded via entry points."""

import importlib.metadata


def test_saluki_plugin_entry_point_smoke():
    """Saluki entry point is present and loadable."""
    eps = importlib.metadata.entry_points(group="biolm.plugins")
    saluki_ep = next((ep for ep in eps if ep.name == "saluki"), None)
    assert saluki_ep is not None, "saluki entry point not found"
    saluki_ep.load()()
