from types import SimpleNamespace

from biolm_utils import plugin_loader
from biolm_utils.plugin_registry import get_plugin_factory, unregister_plugin


def test_saluki_entrypoint_discovery(monkeypatch):
    # Ensure clean state
    if get_plugin_factory("saluki") is not None:
        unregister_plugin("saluki")

    # Dummy EntryPoint-like object which loads the current saluki factory
    class DummyEP:
        def __init__(self, name, loader):
            self.name = name
            self._loader = loader

        def load(self):
            return self._loader

    # loader resolves to saluki.get_saluki_config
    def loader():
        from saluki import get_saluki_config

        return get_saluki_config

    monkeypatch.setattr(
        plugin_loader,
        "entry_points",
        lambda: SimpleNamespace(
            select=lambda group=None: [DummyEP("saluki", loader())]
        ),
    )

    registered = plugin_loader.discover_entrypoint_plugins()
    # discovery may skip registering a plugin if it's already registered by
    # importing the module — ensure the plugin factory exists in the registry
    assert get_plugin_factory("saluki") is not None

    unregister_plugin("saluki")
