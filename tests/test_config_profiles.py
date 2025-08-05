"""Tests for environment-specific configuration profiles."""

import os
import tempfile
from pathlib import Path
import pytest
import yaml

from sdb.config import Settings, load_settings


def test_default_settings():
    """Test that default settings load without any configuration."""
    settings = Settings()
    assert settings.openai_model == "gpt-4"
    assert settings.ui_secret_key == "change-me"
    assert settings.sessions_db == "sessions.db"
    assert settings.tracing is False


def test_development_profile_loading(monkeypatch):
    """Test loading development configuration profile."""
    # Create a temporary config directory
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config"
        config_dir.mkdir()
        
        # Create development.yml
        dev_config = {
            "openai_model": "gpt-3.5-turbo",
            "tracing": True,
            "ui_secret_key": "dev-secret",
            "message_rate_limit": 100
        }
        
        dev_file = config_dir / "development.yml"
        with open(dev_file, "w") as f:
            yaml.dump(dev_config, f)
        
        # Change to temp directory and set environment
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            monkeypatch.setenv("SDBENCH_ENV", "development")
            
            settings = load_settings()
            assert settings.openai_model == "gpt-3.5-turbo"
            assert settings.tracing is True
            assert settings.ui_secret_key == "dev-secret"
            assert settings.message_rate_limit == 100
        finally:
            os.chdir(original_cwd)


def test_local_overrides(monkeypatch):
    """Test that local.yml overrides profile settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config"
        config_dir.mkdir()
        
        # Create development.yml
        dev_config = {
            "openai_model": "gpt-4",
            "ui_budget_limit": 1000.0
        }
        dev_file = config_dir / "development.yml"
        with open(dev_file, "w") as f:
            yaml.dump(dev_config, f)
        
        # Create local.yml override
        local_config = {
            "openai_model": "gpt-3.5-turbo",  # Override
            "tracing": True                   # Additional setting
        }
        local_file = config_dir / "local.yml"
        with open(local_file, "w") as f:
            yaml.dump(local_config, f)
        
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            monkeypatch.setenv("SDBENCH_ENV", "development")
            
            settings = load_settings()
            assert settings.openai_model == "gpt-3.5-turbo"  # From local.yml
            assert settings.ui_budget_limit == 1000.0        # From development.yml
            assert settings.tracing is True                  # From local.yml
        finally:
            os.chdir(original_cwd)


def test_environment_variables_override_all(monkeypatch):
    """Test that environment variables override all configuration files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config"
        config_dir.mkdir()
        
        # Create development.yml
        dev_config = {"openai_model": "gpt-4", "metrics_port": 8000}
        dev_file = config_dir / "development.yml"
        with open(dev_file, "w") as f:
            yaml.dump(dev_config, f)
        
        # Create local.yml
        local_config = {"openai_model": "gpt-3.5-turbo"}
        local_file = config_dir / "local.yml"
        with open(local_file, "w") as f:
            yaml.dump(local_config, f)
        
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            monkeypatch.setenv("SDBENCH_ENV", "development")
            monkeypatch.setenv("OPENAI_MODEL", "claude-3-haiku")
            monkeypatch.setenv("SDB_METRICS_PORT", "9000")
            
            settings = load_settings()
            assert settings.openai_model == "claude-3-haiku"  # From env var
            assert settings.metrics_port == 9000             # From env var
        finally:
            os.chdir(original_cwd)


def test_manual_config_file_loading():
    """Test manually specifying a configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        config = {
            "openai_model": "custom-model",
            "ui_token_ttl": 7200,
            "tracing": True
        }
        yaml.dump(config, f)
        f.flush()
        
        try:
            settings = load_settings(f.name)
            assert settings.openai_model == "custom-model"
            assert settings.ui_token_ttl == 7200
            assert settings.tracing is True
        finally:
            os.unlink(f.name)


def test_nonexistent_profile_uses_defaults(monkeypatch):
    """Test that nonexistent profile falls back to defaults."""
    monkeypatch.setenv("SDBENCH_ENV", "nonexistent")
    settings = load_settings()
    
    # Should use default values since profile doesn't exist
    assert settings.openai_model == "gpt-4"
    assert settings.ui_secret_key == "change-me"


def test_invalid_yaml_file():
    """Test that invalid YAML file raises appropriate error."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write("invalid: yaml: content: [")
        f.flush()
        
        try:
            with pytest.raises(yaml.YAMLError):
                load_settings(f.name)
        finally:
            os.unlink(f.name)


def test_validation_with_profile_config():
    """Test that Pydantic validation works with profile configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        config = {
            "metrics_port": -1,  # Invalid port
            "openai_model": "gpt-4"
        }
        yaml.dump(config, f)
        f.flush()
        
        try:
            with pytest.raises(ValueError, match="metrics_port must be between 1 and 65535"):
                load_settings(f.name)
        finally:
            os.unlink(f.name)