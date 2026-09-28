"""Test doubles. None of these perform network I/O."""

from __future__ import annotations

from orchestrator_worker.aggregators import Aggregator
from orchestrator_worker.context import ExecutionContext
from orchestrator_worker.llm import LLMClient, LLMRequest, LLMResponse
from orchestrator_worker.plan import Plan, PlanStep
from orchestrator_worker.planner import Planner
from orchestrator_worker.workers import BaseWorker, WorkerRegistry


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


class RecordingWorker(BaseWorker):
    """Worker that appends its name to a shared log when it is executed."""

    def __init__(self, name: str, log: list[str], output: str | None = None) -> None:
        self.name = name
        self.log = log
        self.output = output if output is not None else f"{name} output"
        self.calls: list[str] = []

    def execute(self, task: str) -> str:
        self.log.append(self.name)
        self.calls.append(task)
        return self.output


class SpyRegistry(WorkerRegistry):
    """WorkerRegistry that records every get() lookup."""

    def __init__(self) -> None:
        super().__init__()
        self.get_calls: list[str] = []

    def get(self, name: str) -> BaseWorker:
        self.get_calls.append(name)
        return super().get(name)


class SpyPlanner(Planner):
    """Planner that records calls and returns a fixed plan, or raises."""

    def __init__(self, plan: Plan | None = None, error: Exception | None = None) -> None:
        self.plan = plan or Plan(
            goal="a goal",
            steps=[PlanStep(id="step_1", task="a task", worker_name="research")],
        )
        self.error = error
        self.calls: list[str] = []

    def create_plan(self, task: str) -> Plan:
        self.calls.append(task)
        if self.error is not None:
            raise self.error
        return self.plan


class RecordingAggregator(Aggregator):
    """Aggregator double that records contexts and returns a fixed answer."""

    def __init__(self, answer: str = "final answer", error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.calls: list[ExecutionContext] = []

    def aggregate(self, context: ExecutionContext) -> str:
        self.calls.append(context)
        if self.error is not None:
            raise self.error
        return self.answer