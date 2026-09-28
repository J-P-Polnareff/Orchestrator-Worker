"""Test doubles. None of these perform network I/O."""

from __future__ import annotations

from orchestrator_worker.llm import LLMClient, LLMRequest, LLMResponse
from orchestrator_worker.workers import BaseWorker


class FakeLLMClient(LLMClient):
    """Records every request and returns a canned response, or raises."""

    name = "fake"

    def __init__(
        self,
        response: LLMResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.requests: list[LLMRequest] = []
        self._response = response
        self._error = error

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        if self._response is None:
            return LLMResponse(content="", model="fake-model")
        return self._response


class StubWorker(BaseWorker):
    """Worker with scripted behaviour, for orchestrator tests."""

    name = "stub"

    def __init__(self, output: str = "stub output", error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.calls: list[str] = []

    def execute(self, task: str) -> str:
        self.calls.append(task)
        if self.error is not None:
            raise self.error
        return self.output


def fake_response(content: str = "stub answer", model: str = "fake-model") -> LLMResponse:
    """Build an LLMResponse without touching any provider."""
    return LLMResponse(content=content, model=model)


class NamedWorker(BaseWorker):
    """Minimal worker whose name is chosen at construction time."""

    def __init__(self, name: str, output: str | None = None) -> None:
        self.name = name
        self.output = output if output is not None else f"{name} output"
        self.calls: list[str] = []

    def execute(self, task: str) -> str:
        self.calls.append(task)
        return self.output