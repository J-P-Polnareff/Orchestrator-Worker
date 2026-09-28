"""Phase 4C tests: the ExecutionContext data model."""

from __future__ import annotations

import ast
import pathlib
from dataclasses import FrozenInstanceError

import pytest

from orchestrator_worker.context import ExecutionContext, ExecutionContextError
from orchestrator_worker.execution import ExecutionResult
from orchestrator_worker.plan import Plan, PlanStep
from orchestrator_worker.state import AgentState


def make_plan(*step_ids: str) -> Plan:
    return Plan(
        goal="a goal",
        steps=[
            PlanStep(id=step_id, task=f"task for {step_id}", worker_name="research")
            for step_id in step_ids
        ],
    )


def make_result(step_id: str) -> ExecutionResult:
    return ExecutionResult(
        step_id=step_id,
        task=f"task for {step_id}",
        worker_name="research",
        state=AgentState(user_task=f"task for {step_id}", final_result="done"),
    )


def make_context(plan: Plan, *step_ids: str) -> ExecutionContext:
    return ExecutionContext(plan=plan, results=tuple(make_result(i) for i in step_ids))


def test_execution_context_stores_the_plan_and_results():
    plan = make_plan("step_1", "step_2")
    results = (make_result("step_1"), make_result("step_2"))

    context = ExecutionContext(plan=plan, results=results)

    assert context.plan is plan
    assert context.results == results


def test_execution_context_results_are_a_tuple():
    plan = make_plan("step_1")
    context = make_context(plan, "step_1")
    assert isinstance(context.results, tuple)


def test_execution_context_is_frozen():
    context = make_context(make_plan("step_1"), "step_1")
    with pytest.raises(FrozenInstanceError):
        context.plan = make_plan("step_2")  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        context.results = ()  # type: ignore[misc]


def test_execution_context_is_a_dataclass_with_the_expected_fields():
    assert ExecutionContext.__dataclass_fields__.keys() == {"plan", "results"}


@pytest.mark.parametrize("value", [None, "a plan", 123, [PlanStep]])
def test_execution_context_rejects_a_plan_that_is_not_a_plan(value):
    with pytest.raises(ExecutionContextError, match="ExecutionContext.plan must be a Plan"):
        ExecutionContext(plan=value, results=())


def test_execution_context_rejects_results_that_are_not_a_tuple():
    with pytest.raises(ExecutionContextError, match="results must be a tuple"):
        ExecutionContext(plan=make_plan("step_1"), results=[make_result("step_1")])


@pytest.mark.parametrize("value", [None, "a result", {"step_id": "step_1"}, 42])
def test_execution_context_rejects_entries_that_are_not_execution_results(value):
    plan = make_plan("step_1")
    with pytest.raises(ExecutionContextError, match=r"results\[0\]"):
        ExecutionContext(plan=plan, results=(value,))


def test_execution_context_accepts_results_that_cover_the_plan():
    plan = make_plan("step_1", "step_2")
    context = make_context(plan, "step_1", "step_2")
    assert [result.step_id for result in context.results] == ["step_1", "step_2"]


def test_execution_context_rejects_missing_steps():
    plan = make_plan("step_1", "step_2")
    with pytest.raises(ExecutionContextError, match="missing steps"):
        make_context(plan, "step_1")


def test_execution_context_rejects_unknown_steps():
    plan = make_plan("step_1", "step_2")
    with pytest.raises(ExecutionContextError, match="not in the plan"):
        make_context(plan, "step_1", "step_3")


def test_execution_context_rejects_duplicate_steps():
    plan = make_plan("step_1", "step_2")
    with pytest.raises(ExecutionContextError, match="duplicate step ids"):
        make_context(plan, "step_1", "step_1", "step_2")


def test_execution_context_rejects_a_plan_with_no_matching_results():
    plan = make_plan("step_1")
    with pytest.raises(ExecutionContextError):
        ExecutionContext(plan=plan, results=())


def test_execution_context_allows_results_in_any_order():
    plan = make_plan("step_1", "step_2", "step_3")

    context = make_context(plan, "step_3", "step_1", "step_2")

    assert [result.step_id for result in context.results] == [
        "step_3",
        "step_1",
        "step_2",
    ]


def test_execution_context_does_not_modify_the_plan():
    plan = make_plan("step_1", "step_2")
    goal_before = plan.goal
    steps_before = list(plan.steps)

    context = make_context(plan, "step_2", "step_1")

    assert context.plan is plan
    assert plan.goal == goal_before
    assert list(plan.steps) == steps_before
    assert [step.id for step in plan.steps] == ["step_1", "step_2"]


def test_execution_context_error_is_a_runtime_error():
    assert issubclass(ExecutionContextError, RuntimeError)


def test_context_module_has_no_llm_worker_or_router_dependency():
    from orchestrator_worker import context as context_module

    tree = ast.parse(pathlib.Path(context_module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))

    assert imported == ["__future__", "dataclasses", ".execution", ".plan"]