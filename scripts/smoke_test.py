r"""Manual, cost-controlled real-DeepSeek end-to-end smoke test.

Run it by hand only:

    .\.venv\Scripts\python.exe scripts\smoke_test.py

It is deliberately NOT a pytest test: it makes a real API call, so it never runs
as part of the suite. It goes through the Phase 5A composition root with
``llm=None``, so ``build_orchestrator()`` builds the configured provider itself,
and then runs the Phase 4F ``run_pipeline()`` with ``max_retries=0``.

It checks the real stage order (planner -> worker(s) -> evaluator ->
aggregator) by wrapping the shared LLM client at runtime; no production module
is modified. The API key is never printed - only whether it is present.
"""

from __future__ import annotations

from orchestrator_worker.application import build_orchestrator
from orchestrator_worker.config import Settings
from orchestrator_worker.llm import LLMClient, LLMRequest, LLMResponse

TASK = (
    "请用中文用两句话解释 Python 是什么，并说明它最常见的两个应用方向。"
    "只需要一次 research 回答即可，不需要写代码，不需要工具，"
    "最终答案保持在两句话左右。"
)

_STAGE_PREFIXES = (
    ("You are a planning component.", "planner"),
    ("You are a research worker.", "research"),
    ("You are a coding worker.", "coding"),
    ("You are an execution-result evaluator.", "evaluator"),
    ("You are an aggregation component.", "aggregator"),
)


def _stage_of(request: LLMRequest) -> str:
    """Name the pipeline stage that issued ``request``, from its system prompt."""
    system = request.messages[0].content if request.messages else ""
    for prefix, stage in _STAGE_PREFIXES:
        if system.startswith(prefix):
            return stage
    return "unknown"


class _RecordingClient(LLMClient):
    """Delegates every call to the real client and records the calling stage."""

    name = "smoke-recording"

    def __init__(self, inner: LLMClient, stages: list[str]) -> None:
        self._inner = inner
        self._stages = stages

    def complete(self, request: LLMRequest) -> LLMResponse:
        self._stages.append(_stage_of(request))
        return self._inner.complete(request)


def _instrument(orchestrator) -> list[str]:
    """Wrap the shared client so the run can report the real stage order.

    All LLM-backed components share one client (Phase 5A), so wrapping it once
    observes every call without changing any behaviour.
    """
    components = [
        orchestrator._planner,
        orchestrator._evaluator,
        orchestrator._aggregator,
        *(orchestrator._registry.get(name) for name in orchestrator._registry.names()),
    ]

    stages: list[str] = []
    proxy = _RecordingClient(components[0]._llm, stages)
    for component in components:
        component._llm = proxy
    return stages


def main() -> int:
    print("Smoke test started")

    settings = Settings.from_env()
    if settings.api_key:
        print("DEEPSEEK_API_KEY is configured.")
    else:
        print("DEEPSEEK_API_KEY is missing.")
        return 1

    print(f"Task: {TASK}")

    orchestrator = build_orchestrator()
    stages = _instrument(orchestrator)

    try:
        answer = orchestrator.run_pipeline(TASK, max_retries=0)
    except Exception as exc:  # report and stop - never catch and retry
        seen = " -> ".join(stages) if stages else "(none)"
        print(f"Observed stages before failure: {seen}")
        print(f"{type(exc).__name__}: {exc}")
        return 1

    print(f"Observed stages: {' -> '.join(stages)}")
    print("Final answer:")
    print(answer)
    print("Smoke test succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())