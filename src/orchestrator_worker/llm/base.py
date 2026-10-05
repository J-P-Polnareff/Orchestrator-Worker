"""Provider-agnostic LLM interface.

Every agent layer (orchestrator, planner, workers, evaluator, aggregator) talks
to a model only through the types in this module. Swapping DeepSeek for Claude
or OpenAI means adding one adapter under ``orchestrator_worker.llm`` and
registering it, instead of editing agent logic.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from ..tools import Tool, ToolCall

Role = Literal["system", "user", "assistant", "tool"]


class LLMError(RuntimeError):
    """Raised when a provider call fails or a provider is misconfigured."""


def _tool_call_to_dict(call: ToolCall) -> dict[str, Any]:
    """Serialize one tool call the way OpenAI-compatible providers expect."""
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(call.arguments),
        },
    }


@dataclass(frozen=True)
class Message:
    """One provider-neutral chat message.

    A plain ``system`` / ``user`` / ``assistant`` message only needs ``role``
    and ``content``. Tool calling adds two shapes:

    * an ``assistant`` message may carry the ``tool_calls`` it requested,
    * a ``tool`` message carries one tool result plus the ``tool_call_id`` it
      answers.

    ``to_dict`` returns the provider-compatible plain dict; it never contains a
    provider SDK object.
    """

    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ("assistant", "system", "tool", "user"):
            raise LLMError(
                "Message.role must be one of ('assistant', 'system', 'tool', "
                f"'user'), got {self.role!r}."
            )
        if not isinstance(self.content, str):
            raise LLMError(
                "Message.content must be a string, "
                f"got {type(self.content).__name__}."
            )
        if not isinstance(self.tool_calls, tuple):
            raise LLMError(
                "Message.tool_calls must be a tuple, "
                f"got {type(self.tool_calls).__name__}."
            )
        for index, call in enumerate(self.tool_calls):
            if not isinstance(call, ToolCall):
                raise LLMError(
                    f"Message.tool_calls[{index}] must be a ToolCall, "
                    f"got {type(call).__name__}."
                )
        if self.tool_calls and self.role != "assistant":
            raise LLMError(
                "Message.tool_calls are only allowed on an assistant message."
            )

        if self.tool_call_id is not None:
            if not isinstance(self.tool_call_id, str) or not self.tool_call_id.strip():
                raise LLMError("Message.tool_call_id must be a non-empty string.")
            if self.role != "tool":
                raise LLMError(
                    "Message.tool_call_id is only allowed on a tool message."
                )
        elif self.role == "tool":
            raise LLMError("A tool message must set tool_call_id.")

    def to_dict(self) -> dict[str, Any]:
        """Return the provider-compatible message dict."""
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                _tool_call_to_dict(call) for call in self.tool_calls
            ]
        if self.role == "tool":
            payload["tool_call_id"] = self.tool_call_id
        return payload

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls("assistant", content)

    @classmethod
    def tool(cls, content: str, tool_call_id: str) -> "Message":
        return cls("tool", content, tool_call_id=tool_call_id)


@dataclass(frozen=True)
class Usage:
    """Token accounting, normalized across providers."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
@dataclass(frozen=True)
class LLMRequest:
    """Provider-independent chat completion request.

    ``json_mode`` asks the adapter for a JSON-object response when the provider
    supports it (DeepSeek: ``response_format={"type": "json_object"}``). The
    caller still owns parsing and validation of the returned text.
    """

    messages: Sequence[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    stop: Sequence[str] | None = None
    json_mode: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    tools: tuple[Tool, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.tools, tuple):
            raise LLMError(
                f"LLMRequest.tools must be a tuple, got {type(self.tools).__name__}."
            )
        for index, tool in enumerate(self.tools):
            if not isinstance(tool, Tool):
                raise LLMError(
                    f"LLMRequest.tools[{index}] must be a Tool, "
                    f"got {type(tool).__name__}."
                )


@dataclass(frozen=True)
class LLMResponse:
    """Normalized completion result.

    ``tool_calls`` holds provider-neutral :class:`~orchestrator_worker.tools.ToolCall`
    objects parsed by the adapter. Nothing here executes a tool.
    """

    content: str
    model: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    raw: Any = None
    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.tool_calls, tuple):
            raise LLMError(
                "LLMResponse.tool_calls must be a tuple, "
                f"got {type(self.tool_calls).__name__}."
            )
        for index, call in enumerate(self.tool_calls):
            if not isinstance(call, ToolCall):
                raise LLMError(
                    f"LLMResponse.tool_calls[{index}] must be a ToolCall, "
                    f"got {type(call).__name__}."
                )


class LLMClient(ABC):
    """Interface every provider adapter must implement."""

    name: str = "base"

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Run one chat completion and return a normalized response."""

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        """Async wrapper. Adapters may override with a native async path."""
        return await asyncio.to_thread(self.complete, request)

    def health_check(self) -> bool:
        """Cheap config sanity check. Performs no network call by default."""
        return True