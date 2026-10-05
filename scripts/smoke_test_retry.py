r"""Manual, cost-controlled real-DeepSeek retry smoke test.

Run it by hand only:

    .\.venv\Scripts\python.exe scripts\smoke_test_retry.py

It is deliberately NOT a pytest test: it makes real API calls, so it never runs
as part of the suite. Like the Phase 5B-1 smoke test it goes through the
Phase 5A composition root (``build_orchestrator()`` with ``llm=None``) and the
Phase 4F ``run_pipeline()``, this time with ``max_retries=1``.

The first real evaluator verdict is forced to ``passed=False`` by a test-side
wrapper, so the retry path is exercised deterministically and the model never
has to fail on its own. The first evaluator call is still a real DeepSeek call -
only its verdict is overridden; the second call is passed through unchanged.

Everything it changes is harness-local:

* the shared recording LLM client (from ``scripts/smoke_test.py``),
* a planner wrapper that refuses multi-step plans,
* an evaluator wrapper (the forced first failure),
* an aggregator wrapper that records the context it received.

No production module is modified. The API key is never printed - only whether
it is present.
"""

from __future__ import annotations

from orchestrator_worker.aggregators import Aggregator
from orchestrator_worker.application import build_orchestrator
from orchestrator_worker.config import Settings
from orchestrator_worker.context import ExecutionContext
from orchestrator_worker.evaluation import EvaluationResult
from orchestrator_worker.evaluators import Evaluator
from orchestrator_worker.plan import Plan
from orchestrator_worker.planner import Planner

# Sibling harness helper in this directory (defines TASK and the stage recorder).
from smoke_test import TASK, _instrument

FORCED_FIRST_REASON = "forced first evaluation failure for retry smoke test"


class HarnessError(RuntimeError):
    """Raised by the harness when a smoke-test precondition is violated."""


class _CountingPlanner(Planner):
    """Harness seam: records the real plan and refuses multi-step plans.

    The planner must stay real - this only counts calls, keeps the plan for
    reporting and aborts before any worker runs if the plan is not one step.
    """

    def __init__(self, inner: Planner) -> None:
        self._inner = inner
        self.plan: Plan | None = None
        self.calls = 0

    def create_plan(self, task: str) -> Plan:
        self.calls += 1
        plan = self._inner.create_plan(task)
        self.plan = plan
        if len(plan.steps) != 1:
            raise HarnessError(
                f"Planner produced unexpected number of steps: {len(plan.steps)}"
            )
        return plan


class _FirstFailureEvaluator(Evaluator):
    """Harness seam: real evaluator, but the first verdict is forced to fail."""

    def __init__(self, inner: Evaluator) -> None:
        self._inner = inner
        self.calls = 0
        self.forced_first = False
        self.real_verdicts: list[EvaluationResult] = []
        self.contexts: list[ExecutionContext] = []

    def evaluate(self, context: ExecutionContext) -> EvaluationResult:
        self.calls += 1
        if self.calls > 2:
            raise HarnessError(
                f"evaluator called {self.calls} times; max_retries=1 allows 2"
            )

        self.contexts.append(context)
        real = self._inner.evaluate(context)  # always a real DeepSeek call
        self.real_verdicts.append(real)

        if self.calls == 1:
            self.forced_first = True
            return EvaluationResult(passed=False, reason=FORCED_FIRST_REASON)
        return real


class _RecordingAggregator(Aggregator):
    """Harness seam: delegates to the real aggregator and records contexts."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.contexts: list[ExecutionContext] = []

    def aggregate(self, context: ExecutionContext) -> str:
        self.contexts.append(context)
        return self._inner.aggregate(context)


def _verify(
    *,
    plan: Plan,
    planner: _CountingPlanner,
    evaluator: _FirstFailureEvaluator,
    aggregator: _RecordingAggregator,
    stages: list[str],
) -> list[str]:
    """Return a list of unmet success conditions (empty means all good)."""
    worker_names = [step.worker_name for step in plan.steps]
    expected = [
        "planner",
        *worker_names,
        "evaluator",
        *worker_names,
        "evaluator",
        "aggregator",
    ]

    problems: list[str] = []
    if planner.calls != 1:
        problems.append(f"planner called {planner.calls} times, expected 1")
    if evaluator.calls != 2:
        problems.append(f"evaluator called {evaluator.calls} times, expected 2")
    if len(aggregator.contexts) != 1:
        problems.append(
            f"aggregator called {len(aggregator.contexts)} times, expected 1"
        )
    if stages != expected:
        problems.append(f"observed stages {stages!r} != expected {expected!r}")
    if not evaluator.forced_first:
        problems.append("first evaluator verdict was not forced to passed=False")

    if len(evaluator.contexts) == 2:
        first, second = evaluator.contexts
        if first is second:
            problems.append("retry reused the first ExecutionContext")
        if first.results is second.results:
            problems.append("retry reused the first results tuple")
        if aggregator.contexts and aggregator.contexts[0] is not second:
            problems.append("aggregator did not receive the second (passing) context")

    return problems


def main() -> int:
    print("Smoke test retry started")

    settings = Settings.from_env()
    if settings.api_key:
        print("DEEPSEEK_API_KEY is configured.")
    else:
        print("DEEPSEEK_API_KEY is missing.")
        return 1

    print(f"Task: {TASK}")

    orchestrator = build_orchestrator()
    stages = _instrument(orchestrator)

    real_planner = orchestrator._planner
    planner = _CountingPlanner(real_planner)
    evaluator = _FirstFailureEvaluator(orchestrator._evaluator)
    aggregator = _RecordingAggregator(orchestrator._aggregator)

    # Runtime-only seams: swap the harness wrappers in after assembly.
    orchestrator._planner = planner
    orchestrator._evaluator = evaluator
    orchestrator._aggregator = aggregator

    try:
        answer = orchestrator.run_pipeline(TASK, max_retries=1)
    except HarnessError as exc:
        print(f"HarnessError: {exc}")
        if planner.plan is not None:
            print(f"Planner step count: {len(planner.plan.steps)}")
            print(
                "Planner workers: "
                + ", ".join(step.worker_name for step in planner.plan.steps)
            )
        print(f"Observed stages: {' -> '.join(stages) if stages else '(none)'}")
        return 1
    except Exception as exc:  # report and stop - never catch and retry
        print(f"Observed stages: {' -> '.join(stages) if stages else '(none)'}")
        print(f"{type(exc).__name__}: {exc}")
        return 1

    assert planner.plan is not None  # set by _CountingPlanner during the run

    print(f"Plan goal: {planner.plan.goal}")
    print(f"Planner step count: {len(planner.plan.steps)}")
    print(
        "Planner workers: "
        + ", ".join(step.worker_name for step in planner.plan.steps)
    )
    print(f"Observed stages: {' -> '.join(stages)}")
    worker_names = [step.worker_name for step in planner.plan.steps]
    worker_calls = sum(stages.count(name) for name in worker_names)
    print(
        "Stage counts: "
        f"planner={stages.count('planner')} "
        f"worker={worker_calls} "
        f"evaluator={stages.count('evaluator')} "
        f"aggregator={stages.count('aggregator')}"
    )
    print(f"Real first evaluator verdict: {evaluator.real_verdicts[0]}")
    print(f"Forced first verdict: passed=False ({FORCED_FIRST_REASON})")
    print(f"Second evaluator verdict: {evaluator.real_verdicts[1]}")
    print(f"Total LLM calls observed: {len(stages)}")

    problems = _verify(
        plan=planner.plan,
        planner=planner,
        evaluator=evaluator,
        aggregator=aggregator,
        stages=stages,
    )
    if problems:
        print("Smoke test retry FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("Final answer:")
    print(answer)
    print("Smoke test retry succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())