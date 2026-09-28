"""Tests for the LLM planner."""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

from fakes import FakeLLMClient, NamedWorker, SpyRegistry, fake_response
from orchestrator_worker.llm import LLMClient, LLMError, LLMRequest, LLMResponse
from orchestrator_worker.plan import Plan, PlanStep
from orchestrator_worker.planner import (
    LLMPlanner,
    PlannerError,
    PlannerLLMError,
    PlanParsingError,
    PlanValidationError,
    UnknownWorkerError,
)
from orchestrator_worker.planner import base as planner_base
from orchestrator_worker.planner import llm as planner_llm

DEFAULT_WORKERS = ("research", "coding")

VALID_PLAN = {
    "goal": "study asyncio and write an example",
    "steps": [
        {"id": "step_1", "task": "research asyncio", "worker_name": "research"},
        {"id": "step_2", "task": "write a small example", "worker_name": "coding"},
    ],
}


def make_planner(payload, *, workers=DEFAULT_WORKERS):
    """Build a planner whose fake LLM returns ``payload`` (dict or raw text)."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    llm = FakeLLMClient(fake_response(text))
    return LLMPlanner(llm, available_workers=list(workers)), llm


def test_valid_json_becomes_a_plan():
    planner, _ = make_planner(VALID_PLAN)
    plan = planner.create_plan("study asyncio and write an example")
    assert isinstance(plan, Plan)
    assert plan.goal == "study asyncio and write an example"


def test_goal_is_preserved_from_the_llm():
    planner, _ = make_planner(VALID_PLAN)
    assert planner.create_plan("anything").goal == "study asyncio and write an example"


def test_multiple_steps_are_parsed_in_order():
    planner, _ = make_planner(VALID_PLAN)
    plan = planner.create_plan("anything")
    assert len(plan.steps) == 2
    assert [step.id for step in plan.steps] == ["step_1", "step_2"]


def test_step_fields_are_preserved():
    planner, _ = make_planner(VALID_PLAN)
    plan = planner.create_plan("anything")
    assert plan.steps[0] == PlanStep(
        id="step_1", task="research asyncio", worker_name="research"
    )
    assert plan.steps[1] == PlanStep(
        id="step_2", task="write a small example", worker_name="coding"
    )


def test_plan_is_a_plain_data_object():
    planner, _ = make_planner(VALID_PLAN)
    plan = planner.create_plan("anything")
    assert isinstance(plan.steps[0], PlanStep)
    assert all(isinstance(step, PlanStep) for step in plan.steps)


def test_planner_uses_the_llm_abstraction():
    class MinimalLLM(LLMClient):
        name = "minimal"

        def __init__(self, text: str) -> None:
            self.text = text
            self.requests: list[LLMRequest] = []

        def complete(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            return LLMResponse(content=self.text, model="minimal")

    llm = MinimalLLM(json.dumps(VALID_PLAN))
    plan = LLMPlanner(llm, available_workers=["research", "coding"]).create_plan("task")
    assert plan.goal == "study asyncio and write an example"
    assert len(llm.requests) == 1


def test_planner_sends_a_system_prompt_and_the_task():
    planner, llm = make_planner(VALID_PLAN)
    planner.create_plan("plan this for me")

    request = llm.requests[0]
    assert request.messages[0].role == "system"
    assert request.messages[1].role == "user"
    assert request.messages[1].content == "plan this for me"


def test_prompt_lists_the_available_workers():
    planner, llm = make_planner(VALID_PLAN, workers=("research", "coding"))
    planner.create_plan("task")
    system_prompt = llm.requests[0].messages[0].content
    assert "- research" in system_prompt
    assert "- coding" in system_prompt


def test_prompt_asks_for_json_and_forbids_markdown():
    planner, llm = make_planner(VALID_PLAN)
    planner.create_plan("task")
    system_prompt = llm.requests[0].messages[0].content.lower()
    assert "json only" in system_prompt
    assert "no markdown" in system_prompt


def test_prompt_forbids_executing_the_task():
    planner, llm = make_planner(VALID_PLAN)
    planner.create_plan("task")
    system_prompt = llm.requests[0].messages[0].content.lower()
    assert "never execute the task" in system_prompt


def test_json_mode_is_requested():
    planner, llm = make_planner(VALID_PLAN)
    planner.create_plan("task")
    assert llm.requests[0].json_mode is True


def test_available_workers_are_sorted_and_deduplicated():
    llm = FakeLLMClient(fake_response(json.dumps(VALID_PLAN)))
    planner = LLMPlanner(llm, available_workers=["coding", "research", "coding"])
    assert planner.available_workers == ("coding", "research")


def test_custom_system_prompt_is_used():
    llm = FakeLLMClient(fake_response(json.dumps(VALID_PLAN)))
    planner = LLMPlanner(
        llm, available_workers=list(DEFAULT_WORKERS), system_prompt="be terse"
    )
    planner.create_plan("task")
    assert llm.requests[0].messages[0].content == "be terse"

# --------------------------------------------------------------------------
# malformed LLM output
# --------------------------------------------------------------------------


@pytest.mark.parametrize("payload", ["", "   ", "\n\t"])
def test_empty_response_is_a_parsing_error(payload):
    planner, _ = make_planner(payload)
    with pytest.raises(PlanParsingError, match="empty response"):
        planner.create_plan("task")


@pytest.mark.parametrize("text", ["{not json", "goal: x", '{"goal": }', "[,]"])
def test_invalid_json_is_a_parsing_error(text):
    planner, _ = make_planner(text)
    with pytest.raises(PlanParsingError, match="invalid JSON"):
        planner.create_plan("task")


@pytest.mark.parametrize("text", ["[]", '[{"id": "step_1"}]', '"a string"', "42", "null", "true"])
def test_non_object_json_is_a_parsing_error(text):
    planner, _ = make_planner(text)
    with pytest.raises(PlanParsingError, match="must be a JSON object"):
        planner.create_plan("task")


def test_missing_goal_is_rejected():
    planner, _ = make_planner({"steps": VALID_PLAN["steps"]})
    with pytest.raises(PlanValidationError, match="'goal'"):
        planner.create_plan("task")


@pytest.mark.parametrize("goal", ["", "   ", 123, None, ["a goal"]])
def test_invalid_goal_is_rejected(goal):
    planner, _ = make_planner({"goal": goal, "steps": VALID_PLAN["steps"]})
    with pytest.raises(PlanValidationError, match="'goal'"):
        planner.create_plan("task")


def test_missing_steps_is_rejected():
    planner, _ = make_planner({"goal": "a goal"})
    with pytest.raises(PlanValidationError, match="'steps'"):
        planner.create_plan("task")


def test_empty_steps_is_rejected():
    planner, _ = make_planner({"goal": "a goal", "steps": []})
    with pytest.raises(PlanValidationError, match="must not be empty"):
        planner.create_plan("task")


@pytest.mark.parametrize("steps", [None, "step_1", 7, {"id": "step_1"}])
def test_non_list_steps_is_rejected(steps):
    planner, _ = make_planner({"goal": "a goal", "steps": steps})
    with pytest.raises(PlanValidationError, match="JSON array"):
        planner.create_plan("task")


@pytest.mark.parametrize("step", ["step_1", 7, None, ["step_1"]])
def test_non_object_step_is_rejected(step):
    planner, _ = make_planner({"goal": "a goal", "steps": [step]})
    with pytest.raises(PlanValidationError, match="must be a JSON object"):
        planner.create_plan("task")


@pytest.mark.parametrize("missing", ["id", "task", "worker_name"])
def test_missing_step_field_is_rejected(missing):
    step = {"id": "step_1", "task": "a task", "worker_name": "research"}
    del step[missing]
    planner, _ = make_planner({"goal": "a goal", "steps": [step]})
    with pytest.raises(PlanValidationError, match=f"'{missing}'"):
        planner.create_plan("task")


@pytest.mark.parametrize("field", ["id", "task", "worker_name"])
@pytest.mark.parametrize("value", ["", "   "])
def test_empty_step_field_is_rejected(field, value):
    step = {"id": "step_1", "task": "a task", "worker_name": "research"}
    step[field] = value
    planner, _ = make_planner({"goal": "a goal", "steps": [step]})
    with pytest.raises(PlanValidationError, match=f"'{field}' must not be empty"):
        planner.create_plan("task")


@pytest.mark.parametrize("field", ["id", "task", "worker_name"])
def test_non_string_step_field_is_rejected(field):
    step = {"id": "step_1", "task": "a task", "worker_name": "research"}
    step[field] = 42
    planner, _ = make_planner({"goal": "a goal", "steps": [step]})
    with pytest.raises(PlanValidationError, match=f"'{field}' must be a string"):
        planner.create_plan("task")


def test_duplicate_step_id_is_rejected():
    steps = [
        {"id": "step_1", "task": "first", "worker_name": "research"},
        {"id": "step_1", "task": "second", "worker_name": "coding"},
    ]
    planner, _ = make_planner({"goal": "a goal", "steps": steps})
    with pytest.raises(PlanValidationError, match="duplicate step id"):
        planner.create_plan("task")


# --------------------------------------------------------------------------
# worker name validation
# --------------------------------------------------------------------------


def test_known_worker_names_are_accepted():
    planner, _ = make_planner(VALID_PLAN, workers=("research", "coding"))
    plan = planner.create_plan("task")
    assert {step.worker_name for step in plan.steps} == {"research", "coding"}


def test_unknown_worker_is_rejected_with_a_diagnosable_message():
    payload = {
        "goal": "a goal",
        "steps": [{"id": "step_1", "task": "a task", "worker_name": "translator"}],
    }
    planner, _ = make_planner(payload, workers=("research", "coding"))

    with pytest.raises(UnknownWorkerError) as excinfo:
        planner.create_plan("task")

    message = str(excinfo.value)
    assert "translator" in message
    assert "coding" in message and "research" in message


def test_unknown_worker_does_not_fall_back_to_another_worker():
    payload = {
        "goal": "a goal",
        "steps": [{"id": "step_1", "task": "a task", "worker_name": "translator"}],
    }
    planner, llm = make_planner(payload, workers=("research", "coding"))

    with pytest.raises(UnknownWorkerError):
        planner.create_plan("task")

    assert len(llm.requests) == 1, "must not retry or silently pick another worker"


def test_unknown_worker_error_is_a_validation_error():
    assert issubclass(UnknownWorkerError, PlanValidationError)


def test_empty_available_workers_rejects_every_worker():
    planner, _ = make_planner(VALID_PLAN, workers=())
    with pytest.raises(UnknownWorkerError, match=r"\(none\)"):
        planner.create_plan("task")


@pytest.mark.parametrize("names", [[""], ["research", "   "], ["research", None], [123]])
def test_blank_worker_names_are_rejected_at_construction(names):
    with pytest.raises(ValueError, match="non-empty strings"):
        LLMPlanner(FakeLLMClient(), available_workers=names)


# --------------------------------------------------------------------------
# LLM errors
# --------------------------------------------------------------------------


def test_llm_error_becomes_a_planner_error():
    llm = FakeLLMClient(error=LLMError("DeepSeek chat completion failed: boom"))
    planner = LLMPlanner(llm, available_workers=["research"])

    with pytest.raises(PlannerLLMError) as excinfo:
        planner.create_plan("task")

    assert "boom" in str(excinfo.value)


def test_llm_error_keeps_the_original_cause():
    original = LLMError("boom")
    llm = FakeLLMClient(error=original)
    planner = LLMPlanner(llm, available_workers=["research"])

    with pytest.raises(PlannerLLMError) as excinfo:
        planner.create_plan("task")

    assert excinfo.value.__cause__ is original


def test_planner_error_hierarchy():
    assert issubclass(PlannerError, RuntimeError)
    assert issubclass(PlannerLLMError, PlannerError)
    assert issubclass(PlanParsingError, PlannerError)
    assert issubclass(PlanValidationError, PlannerError)
    assert issubclass(UnknownWorkerError, PlanValidationError)


@pytest.mark.parametrize("task", ["", "   ", "\n"])
def test_empty_task_is_rejected_without_calling_the_llm(task):
    planner, llm = make_planner(VALID_PLAN)
    with pytest.raises(PlannerError, match="non-empty"):
        planner.create_plan(task)
    assert llm.requests == []


# --------------------------------------------------------------------------
# planning must never execute anything
# --------------------------------------------------------------------------


def test_planner_never_looks_up_a_worker_in_the_registry():
    registry = SpyRegistry()
    registry.register(NamedWorker("research"))
    registry.register(NamedWorker("coding"))

    planner, _ = make_planner(VALID_PLAN, workers=registry.names())
    plan = planner.create_plan("task")

    assert plan.goal
    assert registry.get_calls == [], "planning must not look up workers"


def test_planner_never_executes_a_worker():
    research = NamedWorker("research")
    coding = NamedWorker("coding")
    registry = SpyRegistry()
    registry.register(research)
    registry.register(coding)

    planner, _ = make_planner(VALID_PLAN, workers=registry.names())
    planner.create_plan("task")

    assert research.calls == []
    assert coding.calls == []


def test_llm_planner_imports_only_the_expected_modules():
    tree = ast.parse(pathlib.Path(planner_llm.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))
    assert imported == ["__future__", "json", "typing", "..llm", "..plan", ".base"]


def test_planner_base_imports_only_the_expected_modules():
    tree = ast.parse(pathlib.Path(planner_base.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))
    assert imported == ["__future__", "abc", "..plan"]


@pytest.mark.parametrize("module", [planner_llm, planner_base])
def test_planner_modules_never_reference_workers_or_providers(module):
    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    for forbidden in ("WorkerRegistry", "BaseWorker", "Worker.execute", "DeepSeekClient", "openai"):
        assert forbidden not in source