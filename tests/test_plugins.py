from types import SimpleNamespace

import pytest

from sdb.plugins import PluginInfo, validate_plugins


def test_validate_plugins_returns_info():
    infos = validate_plugins()
    assert any(isinstance(i, PluginInfo) for i in infos)


def test_validate_plugins_missing_fields(monkeypatch):
    class DummyEP:
        def __init__(self):
            self.name = "bad"
            self.value = "badpkg:obj"
            self.group = "dx0.personas"
            self.dist = SimpleNamespace(name="badpkg", version="")

    def fake_entry_points(group=None):
        return [DummyEP()] if group == "dx0.personas" else []

    monkeypatch.setattr(
        validate_plugins.__globals__["metadata"], "entry_points", fake_entry_points
    )
    with pytest.raises(RuntimeError):
        validate_plugins(["dx0.personas"])
