"""Runtime configuration, loaded from environment variables / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is a declared dependency
    load_dotenv = None

DEFAULT_PROVIDER = "deepseek"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


class ConfigError(RuntimeError):
    """Raised when configuration is missing or malformed."""


def _get_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc


def _get_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc
@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the runtime configuration."""

    llm_provider: str = DEFAULT_PROVIDER
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_s: float = 60.0
    max_retries: int = 2

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        load_env_file: bool = True,
    ) -> "Settings":
        """Build settings from the environment, optionally reading .env first."""
        if env is None:
            if load_env_file and load_dotenv is not None:
                load_dotenv(override=False)
            env = os.environ

        provider = (env.get("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
        api_key = (env.get("DEEPSEEK_API_KEY") or "").strip() or None

        return cls(
            llm_provider=provider,
            api_key=api_key,
            base_url=(env.get("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL).strip(),
            model=(env.get("DEEPSEEK_MODEL") or DEFAULT_MODEL).strip(),
            temperature=_get_float(env, "LLM_TEMPERATURE", 0.0),
            max_tokens=_get_int(env, "LLM_MAX_TOKENS", 2048),
            timeout_s=_get_float(env, "LLM_TIMEOUT_S", 60.0),
            max_retries=_get_int(env, "LLM_MAX_RETRIES", 2),
        )

    def require_api_key(self) -> str:
        """Return the API key or fail with an actionable message."""
        if not self.api_key:
            raise ConfigError(
                "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and add "
                "your key, or export the variable in your shell."
            )
        return self.api_key

    def redacted(self) -> dict[str, object]:
        """Config summary that is safe to log."""
        key = self.api_key
        if key and len(key) > 8:
            masked: str | None = f"{key[:4]}...{key[-4:]}"
        else:
            masked = "***" if key else None
        return {
            "llm_provider": self.llm_provider,
            "api_key": masked,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_s": self.timeout_s,
            "max_retries": self.max_retries,
        }