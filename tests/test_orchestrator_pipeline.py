"""Phase 4F tests: Orchestrator.run_pipeline() wires the full flow.

The pipeline is pure orchestration over the existing high-level methods:

    plan -> execute_plan_with_retry -> aggregate

so these tests assert the call order and the call counts across every layer
rather than only the final string.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from fakes import (
    RecordingAggregator,
    RecordingEvaluator,
    RecordingWorker,
    SpyPlanner,
    StubWorker,
)
from orchestrator_worker.aggregators import AggregatorError
from orchestrator_worker.context import ExecutionContext
from orchestrator_worker.evaluation import EvaluationError, EvaluationResult
from orchestrator_worker.evaluators import EvaluatorError
from orchestrator_worker.orchestrator import Orchestrator, OrchestratorError
from orchestrator_worker.plan import Plan, PlanStep
from orchestrator_worker.planner import PlannerError
from orchestrator_worker.retry import RetryError
from orchestrator_worker.workers import WorkerError, WorkerRegistry


def step(step_id: str, task: str, worker_name: str) -> PlanStep:
    return PlanStep(id=step_id, task=task, worker_name=worker_name)


def build_plan(*steps: PlanStep) -> Plan:
    return Plan(goal="a goal", steps=list(steps))


def two_step_plan() -> Plan:
    return build_plan(
        step("step_1", "study asyncio", "research"),
        step("step_2", "write an example", "coding"),
    )


def passed() -> EvaluationResult:
    return EvaluationResult(passed=True, reason="ok")


def failed(reason: str) -> EvaluationResult:
    return EvaluationResult(passed=False, reason=reason)


def build_registry(*workers) -> WorkerRegistry:
    registry = WorkerRegistry()
    for worker in workers:
        registry.register(worker)
    return registry


def build_orchestrator(
    *,
    log: list[str],
    plan: Plan | None = None,
    planner_error: Exception | None = None,
    verdicts: list[EvaluationResult] | None = None,
    evaluator_error: Exception | None = None,
    answer: str = "final answer",
    aggregator_error: Exception | None = None,
    worker_names: tuple[str, ...] = ("research", "coding"),
    orchestrator_cls: type[Orchestrator] = Orchestrator,
):
    """Build a fully wired orchestrator whose doubles share one ordered log."""
    planner = SpyPlanner(plan=plan, error=planner_error, log=log)
    evaluator = RecordingEvaluator(verdicts, error=evaluator_error, log=log)
    aggregator = RecordingAggregator(answer=answer, error=aggregator_error, log=log)
    registry = build_registry(*(RecordingWorker(name, log) for name in worker_names))
    orchestrator = orchestrator_cls(
        registry, planner=planner, evaluator=evaluator, aggregator=aggregator
    )
    return orchestrator, planner, evaluator, aggregator


def test_full_success_flow_returns_the_aggregator_answer():
    log: list[str] = []
    orchestrator, _, _, _ = build_orchestrator(log=log, plan=two_step_plan())

    assert orchestrator.run_pipeline("study asyncio") == "final answer"


def test_call_order_is_plan_execute_evaluate_aggregate():
    log: list[str] = []
    orchestrator, _, _, _ = build_orchestrator(log=log, plan=two_step_plan())

    orchestrator.run_pipeline("study asyncio")

    assert log == ["planner", "research", "coding", "evaluator", "aggregator"]


def test_planner_is_called_once_with_the_user_task():
    log: list[str] = []
    orchestrator, planner, _, _ = build_orchestrator(log=log, plan=two_step_plan())

    orchestrator.run_pipeline("study asyncio")

    assert planner.calls == ["study asyncio"]


def test_max_retries_zero_runs_a_single_attempt():
    log: list[str] = []
    orchestrator, planner, evaluator, aggregator = build_orchestrator(
        log=log, plan=two_step_plan()
    )

    orchestrator.run_pipeline("study asyncio", max_retries=0)

    assert log == ["planner", "research", "coding", "evaluator", "aggregator"]
    assert planner.calls == ["study asyncio"]
    assert len(evaluator.calls) == 1
    assert len(aggregator.calls) == 1


def test_retry_reruns_the_whole_plan_and_aggregates_once():
    log: list[str] = []
    orchestrator, planner, evaluator, aggregator = build_orchestrator(
        log=log, plan=two_step_plan(), verdicts=[failed("again"), passed()]
    )

    assert orchestrator.run_pipeline("study asyncio", max_retries=1) == "final answer"

    assert log == [
        "planner",
        "research",
        "coding",
        "evaluator",
        "research",
        "coding",
        "evaluator",
        "aggregator",
    ]
    assert planner.calls == ["study asyncio"]
    assert len(evaluator.calls) == 2
    assert len(aggregator.calls) == 1


def test_three_attempts_keep_one_plan_one_aggregation():
    log: list[str] = []
    orchestrator, planner, evaluator, aggregator = build_orchestrator(
        log=log,
        plan=two_step_plan(),
        verdicts=[failed("one"), failed("two"), passed()],
    )

    orchestrator.run_pipeline("study asyncio", max_retries=2)

    assert planner.calls == ["study asyncio"]
    assert log.count("research") == 3
    assert log.count("evaluator") == 3
    assert log.count("aggregator") == 1
    assert len(evaluator.calls) == 3
    assert len(aggregator.calls) == 1


def test_aggregator_receives_only_the_context_that_passed_evaluation():
    log: list[str] = []
    orchestrator, _, evaluator, aggregator = build_orchestrator(
        log=log, plan=two_step_plan(), verdicts=[failed("again"), passed()]
    )

    orchestrator.run_pipeline("study asyncio", max_retries=1)

    first, second = evaluator.calls
    assert first is not second
    assert first.plan is second.plan
    assert aggregator.calls == [second]
    assert aggregator.calls[0] is not first


def test_exhausted_retry_raises_without_aggregating():
    log: list[str] = []
    orchestrator, planner, evaluator, aggregator = build_orchestrator(
        log=log, plan=two_step_plan(), verdicts=[failed("one"), failed("two")]
    )

    with pytest.raises(RetryError) as excinfo:
        orchestrator.run_pipeline("study asyncio", max_retries=1)

    assert "two" in str(excinfo.value)
    assert planner.calls == ["study asyncio"]
    assert log == [
        "planner",
        "research",
        "coding",
        "evaluator",
        "research",
        "coding",
        "evaluator",
    ]
    assert len(evaluator.calls) == 2
    assert aggregator.calls == []


def test_planner_failure_skips_worker_evaluator_and_aggregator():
    log: list[str] = []
    error = PlannerError("boom")
    orchestrator, _, evaluator, aggregator = build_orchestrator(
        log=log, planner_error=error
    )

    with pytest.raises(PlannerError) as excinfo:
        orchestrator.run_pipeline("study asyncio")

    assert excinfo.value is error
    assert log == ["planner"]
    assert evaluator.calls == []
    assert aggregator.calls == []


def test_worker_failure_skips_evaluator_and_aggregator():
    log: list[str] = []
    error = WorkerError("boom")
    planner = SpyPlanner(
        plan=build_plan(step("step_1", "study asyncio", "stub")), log=log
    )
    evaluator = RecordingEvaluator(log=log)
    aggregator = RecordingAggregator(log=log)
    worker = StubWorker(error=error)
    orchestrator = Orchestrator(
        build_registry(worker), planner=planner, evaluator=evaluator, aggregator=aggregator
    )

    with pytest.raises(WorkerError) as excinfo:
        orchestrator.run_pipeline("study asyncio")

    assert excinfo.value is error
    assert log == ["planner"]
    assert evaluator.calls == []
    assert aggregator.calls == []


def test_evaluator_failure_is_not_retried_and_skips_aggregation():
    log: list[str] = []
    error = EvaluatorError("boom")
    orchestrator, _, evaluator, aggregator = build_orchestrator(
        log=log, evaluator_error=error, worker_names=("research",)
    )

    with pytest.raises(EvaluatorError) as excinfo:
        orchestrator.run_pipeline("study asyncio", max_retries=3)

    assert excinfo.value is error
    assert log == ["planner", "research", "evaluator"]
    assert len(evaluator.calls) == 1
    assert aggregator.calls == []


def test_evaluation_failure_is_not_retried_and_skips_aggregation():
    log: list[str] = []
    error = EvaluationError("boom")
    orchestrator, _, evaluator, aggregator = build_orchestrator(
        log=log, evaluator_error=error, worker_names=("research",)
    )

    with pytest.raises(EvaluationError) as excinfo:
        orchestrator.run_pipeline("study asyncio", max_retries=3)

    assert excinfo.value is error
    assert len(evaluator.calls) == 1
    assert aggregator.calls == []


def test_aggregator_failure_propagates_unchanged():
    log: list[str] = []
    error = AggregatorError("boom")
    orchestrator, _, _, aggregator = build_orchestrator(
        log=log, aggregator_error=error, worker_names=("research",)
    )

    with pytest.raises(AggregatorError) as excinfo:
        orchestrator.run_pipeline("study asyncio")

    assert excinfo.value is error
    assert len(aggregator.calls) == 1


def test_missing_planner_raises_orchestrator_error():
    orchestrator = Orchestrator(build_registry(RecordingWorker("research", [])))

    with pytest.raises(OrchestratorError, match="no planner configured"):
        orchestrator.run_pipeline("study asyncio")


def test_missing_evaluator_raises_orchestrator_error():
    orchestrator = Orchestrator(
        build_registry(RecordingWorker("research", [])), planner=SpyPlanner()
    )

    with pytest.raises(OrchestratorError, match="no evaluator configured"):
        orchestrator.run_pipeline("study asyncio")


def test_missing_aggregator_raises_orchestrator_error():
    orchestrator = Orchestrator(
        build_registry(RecordingWorker("research", [])),
        planner=SpyPlanner(),
        evaluator=RecordingEvaluator(),
    )

    with pytest.raises(OrchestratorError, match="no aggregator configured"):
        orchestrator.run_pipeline("study asyncio")


def test_invalid_max_retries_is_delegated_not_reimplemented():
    log: list[str] = []
    orchestrator, _, evaluator, aggregator = build_orchestrator(
        log=log, worker_names=("research",)
    )

    with pytest.raises(OrchestratorError, match="max_retries"):
        orchestrator.run_pipeline("study asyncio", max_retries=True)

    assert "research" not in log
    assert evaluator.calls == []
    assert aggregator.calls == []


class _NoRunOrchestrator(Orchestrator):
    """Fails loudly if the pipeline ever falls back to the legacy run()."""

    def run(self, user_task: str, worker_name: str | None = None):
        raise AssertionError("run_pipeline must not call run()")


def test_run_pipeline_does_not_fall_back_to_run():
    log: list[str] = []
    orchestrator, _, _, _ = build_orchestrator(
        log=log, plan=two_step_plan(), orchestrator_cls=_NoRunOrchestrator
    )

    assert orchestrator.run_pipeline("study asyncio") == "final answer"


def test_run_pipeline_does_not_modify_the_plan():
    log: list[str] = []
    orchestrator, planner, _, _ = build_orchestrator(log=log, plan=two_step_plan())
    plan = planner.plan
    steps_before = list(plan.steps)

    orchestrator.run_pipeline("study asyncio", max_retries=2)

    assert plan.goal == "a goal"
    assert list(plan.steps) == steps_before


def test_run_pipeline_leaves_the_legacy_and_layer_apis_intact():
    log: list[str] = []
    orchestrator, planner, _, aggregator = build_orchestrator(
        log=log, worker_names=("research",)
    )

    plan = orchestrator.plan("study asyncio")
    assert plan is planner.plan

    results = orchestrator.execute_plan(plan)
    assert [result.step_id for result in results] == ["step_1"]

    context = orchestrator.execute_plan_context(plan)
    assert isinstance(context, ExecutionContext)
    assert context.plan is plan

    assert isinstance(orchestrator.execute_plan_with_retry(plan), ExecutionContext)
    assert orchestrator.aggregate(context) == "final answer"
    assert aggregator.calls == [context]

    assert orchestrator.run("study asyncio").worker_name == "research"


def test_orchestrator_does_not_import_concrete_pipeline_components():
    from orchestrator_worker import orchestrator as orchestrator_module

    tree = ast.parse(
        pathlib.Path(orchestrator_module.__file__).read_text(encoding="utf-8")
    )
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))

    forbidden = (
        "planner.llm",
        "evaluators.llm",
        "aggregators.llm",
        "llm.deepseek",
        "workers.research",
        "workers.coding",
        "deepseek",
        "openai",
    )
    assert not any(token in name for token in forbidden for name in imported)
    assert ".planner.base" in imported
    assert ".evaluators.base" in imported
    assert ".aggregators.base" in imported