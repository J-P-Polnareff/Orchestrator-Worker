"""Tests for environment-driven configuration."""

from __future__ import annotations

import pytest

from orchestrator_worker.config import ConfigError, Settings


def test_defaults_when_env_is_empty():
    settings = Settings.from_env(env={})
    assert settings.llm_provider == "deepseek"
    assert settings.model == "deepseek-chat"
    assert settings.base_url == "https://api.deepseek.com/v1"
    assert settings.api_key is None
    assert settings.temperature == 0.0
    assert settings.max_tokens == 2048


def test_env_overrides_are_applied():
    settings = Settings.from_env(
        env={
            "LLM_PROVIDER": "DeepSeek",
            "DEEPSEEK_API_KEY": "sk-test-key-1234567890",
            "DEEPSEEK_MODEL": "deepseek-reasoner",
            "LLM_TEMPERATURE": "0.3",
            "LLM_MAX_TOKENS": "512",
        }
    )
    assert settings.llm_provider == "deepseek"
    assert settings.model == "deepseek-reasoner"
    assert settings.temperature == pytest.approx(0.3)
    assert settings.max_tokens == 512


def test_require_api_key_raises_actionable_error():
    settings = Settings.from_env(env={})
    with pytest.raises(ConfigError, match="DEEPSEEK_API_KEY"):
        settings.require_api_key()


def test_require_api_key_returns_key():
    settings = Settings.from_env(env={"DEEPSEEK_API_KEY": "sk-abc"})
    assert settings.require_api_key() == "sk-abc"


def test_malformed_numbers_raise_config_error():
    with pytest.raises(ConfigError, match="LLM_MAX_TOKENS"):
        Settings.from_env(env={"LLM_MAX_TOKENS": "many"})


def test_redacted_hides_the_secret():
    settings = Settings.from_env(env={"DEEPSEEK_API_KEY": "sk-1234567890abcdef"})
    assert "1234567890abcdef" not in str(settings.redacted())