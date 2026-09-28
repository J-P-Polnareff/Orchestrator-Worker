"""Phase 4D tests: Orchestrator.aggregate() delegates to an Aggregator."""

from __future__ import annotations

import ast
import pathlib

import pytest

from fakes import (
    FakeLLMClient,
    NamedWorker,
    RecordingAggregator,
    RecordingWorker,
    SpyPlanner,
    SpyRegistry,
    fake_response,
)
from orchestrator_worker.aggregators import AggregatorError, LLMAggregator
from orchestrator_worker.context import ExecutionContext
from orchestrator_worker.orchestrator import Orchestrator, OrchestratorError
from orchestrator_worker.plan import Plan, PlanStep
from orchestrator_worker.workers import BaseWorker, WorkerRegistry


def step(step_id: str, task: str, worker_name: str) -> PlanStep:
    return PlanStep(id=step_id, task=task, worker_name=worker_name)


def build_plan(*steps: PlanStep) -> Plan:
    return Plan(goal="a goal", steps=list(steps))


def build_registry(*workers: BaseWorker) -> WorkerRegistry:
    registry = WorkerRegistry()
    for worker in workers:
        registry.register(worker)
    return registry


def build_context(orchestrator: Orchestrator) -> ExecutionContext:
    return orchestrator.execute_plan_context(
        build_plan(step("step_1", "study asyncio", "research"))
    )


def test_aggregate_delegates_to_the_injected_aggregator():
    aggregator = RecordingAggregator()
    orchestrator = Orchestrator(
        build_registry(NamedWorker("research")), aggregator=aggregator
    )
    context = build_context(orchestrator)

    orchestrator.aggregate(context)

    assert aggregator.calls == [context]


def test_aggregate_returns_the_aggregator_answer():
    aggregator = RecordingAggregator(answer="the final answer")
    orchestrator = Orchestrator(
        build_registry(NamedWorker("research")), aggregator=aggregator
    )

    assert orchestrator.aggregate(build_context(orchestrator)) == "the final answer"


def test_aggregate_without_an_aggregator_raises_a_clear_error():
    orchestrator = Orchestrator(build_registry(NamedWorker("research")))

    with pytest.raises(OrchestratorError, match="no aggregator configured"):
        orchestrator.aggregate(build_context(orchestrator))


def test_aggregate_does_not_execute_workers_or_call_the_planner():
    log: list[str] = []
    registry = SpyRegistry()
    registry.register(RecordingWorker("research", log))
    planner = SpyPlanner()
    aggregator = RecordingAggregator()
    orchestrator = Orchestrator(registry, planner=planner, aggregator=aggregator)

    context = build_context(orchestrator)
    log.clear()
    registry.get_calls.clear()
    planner.calls.clear()

    orchestrator.aggregate(context)

    assert log == []
    assert registry.get_calls == []
    assert planner.calls == []
    assert aggregator.calls == [context]


def test_aggregate_propagates_aggregator_errors():
    error = AggregatorError("boom")
    aggregator = RecordingAggregator(error=error)
    orchestrator = Orchestrator(
        build_registry(NamedWorker("research")), aggregator=aggregator
    )

    with pytest.raises(AggregatorError) as excinfo:
        orchestrator.aggregate(build_context(orchestrator))

    assert excinfo.value is error


def test_aggregate_does_not_modify_the_context():
    aggregator = RecordingAggregator()
    orchestrator = Orchestrator(
        build_registry(NamedWorker("research")), aggregator=aggregator
    )
    context = build_context(orchestrator)
    results_before = context.results
    steps_before = list(context.plan.steps)

    orchestrator.aggregate(context)

    assert context.results is results_before
    assert list(context.plan.steps) == steps_before


def test_orchestrator_can_use_an_llm_aggregator_with_a_fake_client():
    llm = FakeLLMClient(fake_response("final answer"))
    orchestrator = Orchestrator(
        build_registry(NamedWorker("research")), aggregator=LLMAggregator(llm)
    )

    assert orchestrator.aggregate(build_context(orchestrator)) == "final answer"
    assert len(llm.requests) == 1


def test_plan_execute_aggregate_chain_with_fakes():
    registry = build_registry(RecordingWorker("research", []))
    planner = SpyPlanner(plan=build_plan(step("step_1", "study asyncio", "research")))
    aggregator = RecordingAggregator(answer="final answer")
    orchestrator = Orchestrator(registry, planner=planner, aggregator=aggregator)

    plan = orchestrator.plan("study asyncio")
    context = orchestrator.execute_plan_context(plan)

    assert orchestrator.aggregate(context) == "final answer"
    assert [result.step_id for result in context.results] == ["step_1"]
    assert planner.calls == ["study asyncio"]


def test_constructions_without_an_aggregator_still_work():
    registry = build_registry(NamedWorker("research"))
    planner = SpyPlanner()

    without_planner = Orchestrator(registry)
    assert without_planner.worker_name == "research"
    assert without_planner.run("task").worker_name == "research"
    assert without_planner.execute_plan(
        build_plan(step("step_1", "task", "research"))
    )[0].step_id == "step_1"

    with_planner = Orchestrator(registry, planner=planner)
    assert with_planner.plan("task") is planner.plan
    assert with_planner.execute_plan_context(
        build_plan(step("step_1", "task", "research"))
    ).plan.goal == "a goal"


def test_orchestrator_depends_on_the_aggregator_abstraction():
    from orchestrator_worker import orchestrator as orchestrator_module

    tree = ast.parse(pathlib.Path(orchestrator_module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))

    assert ".aggregators.base" in imported
    assert not any("aggregators.llm" in name for name in imported)
    assert not any(name.endswith(".llm") for name in imported)