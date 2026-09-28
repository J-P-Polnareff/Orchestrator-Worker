"""Tests for the Plan / PlanStep data model."""

from __future__ import annotations

import ast
import pathlib

import pytest

from orchestrator_worker.plan import Plan, PlanError, PlanStep


def test_create_plan_step():
    step = PlanStep(id="step_1", task="research asyncio", worker_name="research")
    assert step.id == "step_1"
    assert step.task == "research asyncio"
    assert step.worker_name == "research"


@pytest.mark.parametrize("value", ["", "   ", "\n\t"])
def test_plan_step_rejects_empty_id(value):
    with pytest.raises(PlanError, match="PlanStep.id"):
        PlanStep(id=value, task="a task", worker_name="research")


@pytest.mark.parametrize("value", ["", "   "])
def test_plan_step_rejects_empty_task(value):
    with pytest.raises(PlanError, match="PlanStep.task"):
        PlanStep(id="step_1", task=value, worker_name="research")


@pytest.mark.parametrize("value", ["", "   "])
def test_plan_step_rejects_empty_worker_name(value):
    with pytest.raises(PlanError, match="PlanStep.worker_name"):
        PlanStep(id="step_1", task="a task", worker_name=value)


@pytest.mark.parametrize("field", ["id", "task", "worker_name"])
def test_plan_step_rejects_non_string_fields(field):
    values = {"id": "step_1", "task": "a task", "worker_name": "research"}
    values[field] = 123
    with pytest.raises(PlanError, match=f"PlanStep.{field}"):
        PlanStep(**values)


def test_create_plan():
    steps = [
        PlanStep(id="step_1", task="research asyncio", worker_name="research"),
        PlanStep(id="step_2", task="write an example", worker_name="coding"),
    ]
    plan = Plan(goal="study asyncio", steps=steps)
    assert plan.goal == "study asyncio"
    assert plan.steps == steps


@pytest.mark.parametrize("value", ["", "   ", "\n"])
def test_plan_rejects_empty_goal(value):
    steps = [PlanStep(id="step_1", task="a task", worker_name="research")]
    with pytest.raises(PlanError, match="Plan.goal"):
        Plan(goal=value, steps=steps)


def test_plan_rejects_empty_steps():
    with pytest.raises(PlanError, match="must not be empty"):
        Plan(goal="a goal", steps=[])


@pytest.mark.parametrize("steps", [None, "step_1", {}, ("a",)])
def test_plan_rejects_non_list_steps(steps):
    with pytest.raises(PlanError, match="Plan.steps must be a list"):
        Plan(goal="a goal", steps=steps)


def test_plan_rejects_entries_that_are_not_plan_steps():
    with pytest.raises(PlanError, match=r"Plan\.steps\[0\]"):
        Plan(goal="a goal", steps=[{"id": "step_1"}])


def test_plan_step_keeps_its_order():
    first = PlanStep(id="a", task="first", worker_name="research")
    second = PlanStep(id="b", task="second", worker_name="coding")
    plan = Plan(goal="goal", steps=[first, second])
    assert [step.id for step in plan.steps] == ["a", "b"]


def test_plan_error_is_a_runtime_error():
    assert issubclass(PlanError, RuntimeError)


def test_plan_module_has_no_llm_or_worker_dependency():
    from orchestrator_worker import plan as plan_module

    tree = ast.parse(pathlib.Path(plan_module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))

    assert imported == ["__future__", "dataclasses"]