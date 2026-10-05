"""DeepSeek adapter for the provider-agnostic LLM interface.

DeepSeek exposes an OpenAI-compatible chat completions endpoint, so this class
is a thin translation layer over the official ``openai`` SDK. The SDK client is
injectable, which keeps the adapter unit-testable without any network access.

Provider notes:
  * JSON mode requires the literal word "json" to appear in the prompt and is
    not supported by ``deepseek-reasoner``.
  * Reasoning models return chain-of-thought separately; this adapter only
    surfaces the final ``content``.
  * Tool calling uses the non-thinking, OpenAI-compatible function-call flow:
    tool definitions go out as plain JSON-schema dicts and the response's
    ``tool_calls`` are parsed back into provider-neutral ``ToolCall`` objects.
    Nothing is executed here.
"""

from __future__ import annotations

import json
from typing import Any

from ..config import DEFAULT_BASE_URL, DEFAULT_MODEL
from ..tools import Tool, ToolCall
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
        if request.tools:
            payload["tools"] = [_tool_payload(tool) for tool in request.tools]

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
            tool_calls=_tool_calls_from_message(message),
        )


def _usage_from_raw(usage: Any | None) -> Usage:
    if usage is None:
        return Usage()
    return Usage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
    )


def _tool_payload(tool: Tool) -> dict[str, Any]:
    """Serialize a Tool definition for an OpenAI-compatible request.

    Only the public definition is sent; ``tool.handler`` never leaves the
    process.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _tool_calls_from_message(message: Any) -> tuple[ToolCall, ...]:
    raw_calls = getattr(message, "tool_calls", None) or ()
    return tuple(
        _tool_call_from_raw(raw_call, index)
        for index, raw_call in enumerate(raw_calls)
    )


def _tool_call_from_raw(raw_call: Any, index: int) -> ToolCall:
    """Parse one provider tool call into a provider-neutral ``ToolCall``.

    The generated ``arguments`` string is not guaranteed to be valid JSON, so it
    is parsed and required to be a JSON object. Anything else fails loudly:
    there is no defaulting, guessing or silent repair.
    """
    call_id = getattr(raw_call, "id", None)
    if not isinstance(call_id, str) or not call_id.strip():
        raise LLMError(f"DeepSeek tool call #{index} is missing a valid id.")

    call_type = getattr(raw_call, "type", None)
    if call_type != "function":
        raise LLMError(
            f"DeepSeek tool call #{index} has unsupported type {call_type!r}; "
            "only 'function' is supported."
        )

    function = getattr(raw_call, "function", None)
    if function is None:
        raise LLMError(f"DeepSeek tool call #{index} is missing its function.")

    name = getattr(function, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise LLMError(f"DeepSeek tool call #{index} is missing a valid name.")

    raw_arguments = getattr(function, "arguments", None)
    if not isinstance(raw_arguments, str) or not raw_arguments.strip():
        raise LLMError(f"DeepSeek tool call #{index} is missing its arguments.")

    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"DeepSeek tool call #{index} returned malformed JSON arguments: {exc}"
        ) from exc

    if not isinstance(arguments, dict):
        raise LLMError(
            f"DeepSeek tool call #{index} arguments must be a JSON object, "
            f"got {type(arguments).__name__}."
        )

    return ToolCall(id=call_id, name=name, arguments=arguments)