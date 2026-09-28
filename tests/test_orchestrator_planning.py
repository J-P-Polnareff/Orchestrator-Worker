"""Phase 3 tests: Orchestrator planning API (planning only, no execution)."""

from __future__ import annotations

import ast
import pathlib

import pytest

from fakes import FakeLLMClient, NamedWorker, SpyRegistry, StubWorker, fake_response
from orchestrator_worker.orchestrator import Orchestrator, OrchestratorError
from orchestrator_worker.plan import Plan, PlanStep
from orchestrator_worker.planner import LLMPlanner, Planner, PlannerError
from orchestrator_worker.workers import WorkerRegistry


class RecordingPlanner(Planner):
    """Planner double that records calls and returns a fixed plan."""

    def __init__(self, plan: Plan | None = None) -> None:
        self.plan_result = plan or Plan(
            goal="a goal",
            steps=[PlanStep(id="step_1", task="a task", worker_name="research")],
        )
        self.calls: list[str] = []

    def create_plan(self, task: str) -> Plan:
        self.calls.append(task)
        return self.plan_result


def build_registry() -> tuple[SpyRegistry, NamedWorker, NamedWorker]:
    research = NamedWorker("research")
    coding = NamedWorker("coding")
    registry = SpyRegistry()
    registry.register(research)
    registry.register(coding)
    return registry, research, coding


def test_plan_delegates_to_the_injected_planner():
    planner = RecordingPlanner()
    registry, _, _ = build_registry()
    orchestrator = Orchestrator(registry, planner=planner)

    orchestrator.plan("plan this")

    assert planner.calls == ["plan this"]


def test_plan_returns_the_planner_result():
    expected = Plan(
        goal="study asyncio",
        steps=[PlanStep(id="step_1", task="research asyncio", worker_name="research")],
    )
    registry, _, _ = build_registry()
    orchestrator = Orchestrator(registry, planner=RecordingPlanner(expected))

    plan = orchestrator.plan("study asyncio")

    assert plan is expected
    assert plan.goal == "study asyncio"


def test_plan_without_a_planner_raises_a_clear_error():
    registry, _, _ = build_registry()
    with pytest.raises(OrchestratorError, match="no planner configured"):
        Orchestrator(registry).plan("task")


def test_planning_does_not_look_up_or_execute_workers():
    registry, research, coding = build_registry()
    orchestrator = Orchestrator(registry, planner=RecordingPlanner())

    orchestrator.plan("task")

    assert registry.get_calls == []
    assert research.calls == []
    assert coding.calls == []


def test_planning_does_not_execute_workers_end_to_end():
    registry, research, coding = build_registry()
    llm = FakeLLMClient(
        fake_response(
            '{"goal": "g", "steps": ['
            '{"id": "step_1", "task": "t", "worker_name": "research"},'
            '{"id": "step_2", "task": "u", "worker_name": "coding"}]}'
        )
    )
    planner = LLMPlanner(llm, available_workers=registry.names())
    orchestrator = Orchestrator(registry, planner=planner)

    plan = orchestrator.plan("do something")

    assert [step.worker_name for step in plan.steps] == ["research", "coding"]
    assert registry.get_calls == []
    assert research.calls == []
    assert coding.calls == []


def test_planner_errors_propagate_from_plan():
    registry, _, _ = build_registry()
    llm = FakeLLMClient(fake_response("not json at all"))
    orchestrator = Orchestrator(registry, planner=LLMPlanner(llm, ["research"]))

    with pytest.raises(PlannerError):
        orchestrator.plan("task")


def test_run_still_executes_workers_after_phase_3():
    registry = WorkerRegistry()
    registry.register(StubWorker(output="stub answer"))
    orchestrator = Orchestrator(registry, planner=RecordingPlanner())

    state = orchestrator.run("task", worker_name="stub")

    assert state.worker_name == "stub"
    assert state.final_result == "stub answer"


def test_run_defaults_to_research_after_phase_3():
    registry, _, _ = build_registry()
    orchestrator = Orchestrator(registry, planner=RecordingPlanner())

    state = orchestrator.run("task")

    assert state.worker_name == "research"
    assert state.final_result == "research output"


def test_run_does_not_use_the_planner():
    planner = RecordingPlanner()
    registry, _, _ = build_registry()
    orchestrator = Orchestrator(registry, planner=planner)

    orchestrator.run("task")

    assert planner.calls == []


def test_phase2_single_worker_shorthand_still_works_with_a_planner():
    orchestrator = Orchestrator(
        StubWorker(output="stub answer"), planner=RecordingPlanner()
    )
    assert orchestrator.run("task").final_result == "stub answer"
    assert orchestrator.plan("task").goal == "a goal"


def test_orchestrator_does_not_import_the_concrete_planner():
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

    assert imported == [
        "__future__",
        ".plan",
        ".planner.base",
        ".state",
        ".workers.base",
        ".workers.registry",
    ]
    assert not any("planner.llm" in name or "llm_planner" in name for name in imported)
    assert not any("research" in name or "coding" in name for name in imported)