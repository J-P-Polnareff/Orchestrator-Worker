"""DeepSeek adapter for the provider-agnostic LLM interface.

DeepSeek exposes an OpenAI-compatible chat completions endpoint, so this class
is a thin translation layer over the official ``openai`` SDK. The SDK client is
injectable, which keeps the adapter unit-testable without any network access.

Provider notes:
  * JSON mode requires the literal word "json" to appear in the prompt and is
    not supported by ``deepseek-reasoner``.
  * Reasoning models return chain-of-thought separately; this adapter only
    surfaces the final ``content``.
"""

from __future__ import annotations

from typing import Any

from ..config import DEFAULT_BASE_URL, DEFAULT_MODEL
from .base import LLMClient, LLMError, LLMRequest, LLMResponse, Usage


class DeepSeekClient(LLMClient):
    """LLMClient backed by the DeepSeek OpenAI-compatible API."""

    name = "deepseek"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url

        if client is None:
            if not api_key:
                raise LLMError(
                    "DeepSeek API key is missing. Set DEEPSEEK_API_KEY in .env "
                    "or pass api_key explicitly."
                )
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise LLMError(
                    "The 'openai' package is required for the DeepSeek adapter. "
                    "Install it with: pip install -e ."
                ) from exc
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout_s,
                max_retries=max_retries,
            )

        self._client = client
    def complete(self, request: LLMRequest) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_dict() for message in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop is not None:
            payload["stop"] = list(request.stop)
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            raw = self._client.chat.completions.create(**payload)
        except Exception as exc:
            raise LLMError(f"DeepSeek chat completion failed: {exc}") from exc

        if not getattr(raw, "choices", None):
            raise LLMError("DeepSeek returned no choices.")

        choice = raw.choices[0]
        message = getattr(choice, "message", None)
        return LLMResponse(
            content=(getattr(message, "content", None) or ""),
            model=getattr(raw, "model", self.model) or self.model,
            usage=_usage_from_raw(getattr(raw, "usage", None)),
            finish_reason=getattr(choice, "finish_reason", None),
            raw=raw,
        )


def _usage_from_raw(usage: Any | None) -> Usage:
    if usage is None:
        return Usage()
    return Usage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
    )