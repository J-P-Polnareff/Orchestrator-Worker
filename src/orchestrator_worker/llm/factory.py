"""Provider registry: turn a configured provider name into an LLMClient.

Adding a provider (for example Claude) is: write the adapter, then call
``register_provider("anthropic", build_anthropic)``. Agent code never changes.
"""

from __future__ import annotations

from typing import Callable

from ..config import Settings
from .base import LLMClient, LLMError
from .deepseek import DeepSeekClient

ProviderBuilder = Callable[[Settings], LLMClient]

_REGISTRY: dict[str, ProviderBuilder] = {}


def register_provider(name: str, builder: ProviderBuilder) -> None:
    """Register (or replace) a provider builder under a lowercase name."""
    _REGISTRY[name.strip().lower()] = builder


def available_providers() -> list[str]:
    """Sorted list of registered provider names."""
    return sorted(_REGISTRY)


def build_llm_client(settings: Settings | None = None) -> LLMClient:
    """Build the LLM client selected by ``settings.llm_provider``."""
    settings = settings or Settings.from_env()
    key = settings.llm_provider.strip().lower()

    builder = _REGISTRY.get(key)
    if builder is None:
        known = ", ".join(available_providers()) or "none"
        raise LLMError(
            f"Unknown LLM provider {settings.llm_provider!r}. Registered providers: {known}"
        )
    return builder(settings)


def _build_deepseek(settings: Settings) -> LLMClient:
    return DeepSeekClient(
        settings.api_key,
        model=settings.model,
        base_url=settings.base_url,
        timeout_s=settings.timeout_s,
        max_retries=settings.max_retries,
    )


register_provider("deepseek", _build_deepseek)