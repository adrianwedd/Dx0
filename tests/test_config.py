import yaml
import pytest

from sdb.config import load_settings


def test_invalid_metrics_port(tmp_path):
    cfg_path = tmp_path / "settings.yml"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump({"metrics_port": 70000}, fh)
    with pytest.raises(ValueError):
        load_settings(str(cfg_path))


def test_invalid_base_url(tmp_path):
    cfg_path = tmp_path / "settings.yml"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump({"ollama_base_url": "not a url"}, fh)
    with pytest.raises(ValueError):
        load_settings(str(cfg_path))
