"""Phase 4E tests: evaluator-driven whole-plan retry."""

from __future__ import annotations

import ast
import pathlib

import pytest

from fakes import (
    NamedWorker,
    RecordingAggregator,
    RecordingEvaluator,
    RecordingWorker,
    SpyPlanner,
    SpyRegistry,
    StubWorker,
)
from orchestrator_worker.context import ExecutionContext
from orchestrator_worker.evaluation import EvaluationError, EvaluationResult
from orchestrator_worker.evaluators import EvaluatorError
from orchestrator_worker.orchestrator import Orchestrator, OrchestratorError
from orchestrator_worker.plan import Plan, PlanStep
from orchestrator_worker.retry import RetryError
from orchestrator_worker.workers import RegistryError, WorkerError, WorkerRegistry


def step(step_id: str, task: str, worker_name: str) -> PlanStep:
    return PlanStep(id=step_id, task=task, worker_name=worker_name)


def build_plan(*steps: PlanStep) -> Plan:
    return Plan(goal="a goal", steps=list(steps))


def build_registry(*workers) -> WorkerRegistry:
    registry = WorkerRegistry()
    for worker in workers:
        registry.register(worker)
    return registry


def failed(reason: str) -> EvaluationResult:
    return EvaluationResult(passed=False, reason=reason)


def test_passing_evaluation_executes_and_evaluates_once():
    log: list[str] = []
    evaluator = RecordingEvaluator()
    orchestrator = Orchestrator(
        build_registry(RecordingWorker("research", log)), evaluator=evaluator
    )
    plan = build_plan(step("step_1", "study asyncio", "research"))

    context = orchestrator.execute_plan_with_retry(plan)

    assert isinstance(context, ExecutionContext)
    assert log == ["research"]
    assert [result.step_id for result in context.results] == ["step_1"]
    assert len(evaluator.calls) == 1


def test_default_max_retries_allows_one_extra_attempt():
    log: list[str] = []
    evaluator = RecordingEvaluator([failed("try again"), EvaluationResult(True, "ok")])
    orchestrator = Orchestrator(
        build_registry(RecordingWorker("research", log)), evaluator=evaluator
    )
    plan = build_plan(step("step_1", "study asyncio", "research"))

    orchestrator.execute_plan_with_retry(plan)

    assert log == ["research", "research"]
    assert len(evaluator.calls) == 2


def test_max_retries_zero_fails_after_a_single_attempt():
    log: list[str] = []
    evaluator = RecordingEvaluator([failed("not enough")])
    orchestrator = Orchestrator(
        build_registry(RecordingWorker("research", log)), evaluator=evaluator
    )
    plan = build_plan(step("step_1", "study asyncio", "research"))

    with pytest.raises(RetryError) as excinfo:
        orchestrator.execute_plan_with_retry(plan, max_retries=0)

    assert log == ["research"]
    assert len(evaluator.calls) == 1
    assert "after 1 attempt(s)" in str(excinfo.value)
    assert "not enough" in str(excinfo.value)


def test_retry_until_pass_with_max_retries_one():
    log: list[str] = []
    evaluator = RecordingEvaluator(
        [failed("first try failed"), EvaluationResult(True, "second try worked")]
    )
    orchestrator = Orchestrator(
        build_registry(RecordingWorker("research", log)), evaluator=evaluator
    )
    plan = build_plan(step("step_1", "study asyncio", "research"))

    context = orchestrator.execute_plan_with_retry(plan, max_retries=1)

    assert log == ["research", "research"]
    assert len(evaluator.calls) == 2
    assert context is evaluator.calls[1]


def test_max_retries_two_allows_three_attempts():
    log: list[str] = []
    evaluator = RecordingEvaluator(
        [failed("one"), failed("two"), EvaluationResult(True, "three")]
    )
    orchestrator = Orchestrator(
        build_registry(RecordingWorker("research", log)), evaluator=evaluator
    )
    plan = build_plan(step("step_1", "study asyncio", "research"))

    context = orchestrator.execute_plan_with_retry(plan, max_retries=2)

    assert log == ["research", "research", "research"]
    assert len(evaluator.calls) == 3
    assert context is evaluator.calls[2]


def test_exhausted_budget_raises_retry_error_with_the_last_reason():
    log: list[str] = []
    evaluator = RecordingEvaluator([failed("one"), failed("two"), failed("three")])
    orchestrator = Orchestrator(
        build_registry(RecordingWorker("research", log)), evaluator=evaluator
    )
    plan = build_plan(step("step_1", "study asyncio", "research"))

    with pytest.raises(RetryError) as excinfo:
        orchestrator.execute_plan_with_retry(plan, max_retries=2)

    assert log == ["research", "research", "research"]
    assert len(evaluator.calls) == 3
    assert "after 3 attempt(s)" in str(excinfo.value)
    assert "three" in str(excinfo.value)


@pytest.mark.parametrize("value", [1.5, "1", None, [], {}, 1.0])
def test_max_retries_must_be_an_int(value):
    log: list[str] = []
    evaluator = RecordingEvaluator()
    orchestrator = Orchestrator(
        build_registry(RecordingWorker("research", log)), evaluator=evaluator
    )
    plan = build_plan(step("step_1", "study asyncio", "research"))

    with pytest.raises(OrchestratorError, match="max_retries"):
        orchestrator.execute_plan_with_retry(plan, max_retries=value)

    assert log == []
    assert evaluator.calls == []


@pytest.mark.parametrize("value", [True, False])
def test_max_retries_rejects_bools(value):
    log: list[str] = []
    evaluator = RecordingEvaluator()
    orchestrator = Orchestrator(
        build_registry(RecordingWorker("research", log)), evaluator=evaluator
    )
    plan = build_plan(step("step_1", "study asyncio", "research"))

    with pytest.raises(OrchestratorError, match="bools are not accepted"):
        orchestrator.execute_plan_with_retry(plan, max_retries=value)

    assert log == []
    assert evaluator.calls == []


@pytest.mark.parametrize("value", [-1, -5])
def test_max_retries_must_not_be_negative(value):
    log: list[str] = []
    evaluator = RecordingEvaluator()
    orchestrator = Orchestrator(
        build_registry(RecordingWorker("research", log)), evaluator=evaluator
    )
    plan = build_plan(step("step_1", "study asyncio", "research"))

    with pytest.raises(OrchestratorError, match="must be >= 0"):
        orchestrator.execute_plan_with_retry(plan, max_retries=value)

    assert log == []
    assert evaluator.calls == []


@pytest.mark.parametrize("value", [None, "a plan", [], 42])
def test_plan_must_be_a_plan(value):
    log: list[str] = []
    evaluator = RecordingEvaluator()
    orchestrator = Orchestrator(
        build_registry(RecordingWorker("research", log)), evaluator=evaluator
    )

    with pytest.raises(OrchestratorError, match="plan must be a Plan"):
        orchestrator.execute_plan_with_retry(value)

    assert log == []
    assert evaluator.calls == []


def test_missing_evaluator_raises_a_clear_error():
    log: list[str] = []
    orchestrator = Orchestrator(build_registry(RecordingWorker("research", log)))
    plan = build_plan(step("step_1", "study asyncio", "research"))

    with pytest.raises(OrchestratorError, match="no evaluator configured"):
        orchestrator.execute_plan_with_retry(plan)

    assert log == []


def test_evaluator_errors_propagate_without_retry():
    log: list[str] = []
    error = EvaluatorError("boom")
    evaluator = RecordingEvaluator(error=error)
    orchestrator = Orchestrator(
        build_registry(RecordingWorker("research", log)), evaluator=evaluator
    )
    plan = build_plan(step("step_1", "study asyncio", "research"))

    with pytest.raises(EvaluatorError) as excinfo:
        orchestrator.execute_plan_with_retry(plan, max_retries=3)

    assert excinfo.value is error
    assert log == ["research"]
    assert len(evaluator.calls) == 1


def test_evaluation_errors_propagate_without_retry():
    log: list[str] = []
    error = EvaluationError("bad verdict")
    evaluator = RecordingEvaluator(error=error)
    orchestrator = Orchestrator(
        build_registry(RecordingWorker("research", log)), evaluator=evaluator
    )
    plan = build_plan(step("step_1", "study asyncio", "research"))

    with pytest.raises(EvaluationError):
        orchestrator.execute_plan_with_retry(plan, max_retries=3)

    assert log == ["research"]
    assert len(evaluator.calls) == 1


def test_worker_errors_propagate_without_retry():
    evaluator = RecordingEvaluator([failed("never reached")])
    failing = StubWorker(error=WorkerError("boom"))
    orchestrator = Orchestrator(build_registry(failing), evaluator=evaluator)
    plan = build_plan(step("step_1", "study asyncio", "stub"))

    with pytest.raises(WorkerError) as excinfo:
        orchestrator.execute_plan_with_retry(plan, max_retries=3)

    assert str(excinfo.value) == "boom"
    assert evaluator.calls == []


def test_registry_errors_propagate_without_retry():
    evaluator = RecordingEvaluator([failed("never reached")])
    orchestrator = Orchestrator(
        build_registry(NamedWorker("research")), evaluator=evaluator
    )
    plan = build_plan(step("step_1", "study asyncio", "translate"))

    with pytest.raises(RegistryError) as excinfo:
        orchestrator.execute_plan_with_retry(plan, max_retries=3)

    assert type(excinfo.value) is RegistryError
    assert evaluator.calls == []


def test_every_attempt_reruns_every_step():
    log: list[str] = []
    research = RecordingWorker("research", log)
    coding = RecordingWorker("coding", log)
    evaluator = RecordingEvaluator([failed("again"), EvaluationResult(True, "ok")])
    orchestrator = Orchestrator(
        build_registry(research, coding), evaluator=evaluator
    )
    plan = build_plan(
        step("step_1", "study asyncio", "research"),
        step("step_2", "write an example", "coding"),
    )

    orchestrator.execute_plan_with_retry(plan, max_retries=1)

    assert log == ["research", "coding", "research", "coding"]
    assert research.calls == ["study asyncio", "study asyncio"]
    assert coding.calls == ["write an example", "write an example"]


def test_retry_does_not_call_the_planner_and_keeps_the_plan():
    log: list[str] = []
    planner = SpyPlanner()
    evaluator = RecordingEvaluator([failed("again"), EvaluationResult(True, "ok")])
    orchestrator = Orchestrator(
        build_registry(RecordingWorker("research", log)),
        planner=planner,
        evaluator=evaluator,
    )
    plan = build_plan(step("step_1", "study asyncio", "research"))
    goal_before = plan.goal
    steps_before = list(plan.steps)

    result = orchestrator.execute_plan_with_retry(plan, max_retries=1)

    assert planner.calls == []
    assert result.plan is plan
    assert plan.goal == goal_before
    assert list(plan.steps) == steps_before


def test_each_attempt_builds_fresh_contexts_results_and_states():
    log: list[str] = []
    evaluator = RecordingEvaluator([failed("again"), EvaluationResult(True, "ok")])
    orchestrator = Orchestrator(
        build_registry(RecordingWorker("research", log)), evaluator=evaluator
    )
    plan = build_plan(step("step_1", "study asyncio", "research"))

    orchestrator.execute_plan_with_retry(plan, max_retries=1)

    first, second = evaluator.calls
    assert first is not second
    assert first.results is not second.results
    assert first.results[0] is not second.results[0]
    assert first.results[0].state is not second.results[0].state


def test_retry_reuses_the_registry_lookup_per_attempt():
    log: list[str] = []
    registry = SpyRegistry()
    registry.register(RecordingWorker("research", log))
    evaluator = RecordingEvaluator([failed("again"), EvaluationResult(True, "ok")])
    orchestrator = Orchestrator(registry, evaluator=evaluator)
    plan = build_plan(step("step_1", "study asyncio", "research"))

    orchestrator.execute_plan_with_retry(plan, max_retries=1)

    assert registry.get_calls == ["research", "research"]


def test_existing_apis_are_unchanged():
    log: list[str] = []
    registry = build_registry(RecordingWorker("research", log))
    planner = SpyPlanner()
    aggregator = RecordingAggregator(answer="final answer")
    orchestrator = Orchestrator(
        registry, planner=planner, aggregator=aggregator
    )
    plan = build_plan(step("step_1", "study asyncio", "research"))

    results = orchestrator.execute_plan(plan)
    assert isinstance(results, list)
    assert [result.step_id for result in results] == ["step_1"]

    context = orchestrator.execute_plan_context(plan)
    assert isinstance(context, ExecutionContext)

    assert orchestrator.aggregate(context) == "final answer"

    assert orchestrator.plan("a task") is planner.plan
    assert orchestrator.run("a task").worker_name == "research"


def test_orchestrator_does_not_import_the_llm_evaluator():
    from orchestrator_worker import orchestrator as orchestrator_module

    tree = ast.parse(pathlib.Path(orchestrator_module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))

    assert ".evaluators.base" in imported
    assert not any("evaluators.llm" in name for name in imported)
    assert not any("deepseek" in name or "openai" in name for name in imported)