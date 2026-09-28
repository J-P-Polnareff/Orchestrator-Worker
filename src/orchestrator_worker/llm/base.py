"""Provider-agnostic LLM interface.

Every agent layer (orchestrator, planner, workers, evaluator, aggregator) talks
to a model only through the types in this module. Swapping DeepSeek for Claude
or OpenAI means adding one adapter under ``orchestrator_worker.llm`` and
registering it, instead of editing agent logic.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

Role = Literal["system", "user", "assistant"]


class LLMError(RuntimeError):
    """Raised when a provider call fails or a provider is misconfigured."""


@dataclass(frozen=True)
class Message:
    """One chat message."""

    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls("assistant", content)


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


@dataclass(frozen=True)
class LLMResponse:
    """Normalized completion result."""

    content: str
    model: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    raw: Any = None


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