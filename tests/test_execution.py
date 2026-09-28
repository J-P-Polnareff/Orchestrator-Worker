"""Phase 4B tests: the ExecutionResult data model."""

from __future__ import annotations

import ast
import pathlib
from dataclasses import FrozenInstanceError

import pytest

from orchestrator_worker.execution import ExecutionError, ExecutionResult
from orchestrator_worker.state import AgentState


def make_state() -> AgentState:
    return AgentState(
        user_task="study asyncio",
        worker_name="research",
        worker_input="study asyncio",
        worker_output="research answer",
        final_result="research answer",
    )


def make_result(**overrides: object) -> ExecutionResult:
    values: dict[str, object] = {
        "step_id": "step_1",
        "task": "study asyncio",
        "worker_name": "research",
        "state": make_state(),
    }
    values.update(overrides)
    return ExecutionResult(**values)


def test_execution_result_stores_every_field():
    state = make_state()
    result = ExecutionResult(
        step_id="step_1",
        task="study asyncio",
        worker_name="research",
        state=state,
    )

    assert result.step_id == "step_1"
    assert result.task == "study asyncio"
    assert result.worker_name == "research"
    assert result.state is state


def test_execution_result_is_frozen():
    result = make_result()
    with pytest.raises(FrozenInstanceError):
        result.step_id = "step_2"  # type: ignore[misc]


def test_execution_result_is_a_dataclass_with_the_expected_fields():
    assert ExecutionResult.__dataclass_fields__.keys() == {
        "step_id",
        "task",
        "worker_name",
        "state",
    }


@pytest.mark.parametrize("value", ["", "   ", "\n\t"])
def test_execution_result_rejects_empty_step_id(value):
    with pytest.raises(ExecutionError, match="ExecutionResult.step_id"):
        make_result(step_id=value)


@pytest.mark.parametrize("value", ["", "   "])
def test_execution_result_rejects_empty_task(value):
    with pytest.raises(ExecutionError, match="ExecutionResult.task"):
        make_result(task=value)


@pytest.mark.parametrize("value", ["", "   "])
def test_execution_result_rejects_empty_worker_name(value):
    with pytest.raises(ExecutionError, match="ExecutionResult.worker_name"):
        make_result(worker_name=value)


@pytest.mark.parametrize("field", ["step_id", "task", "worker_name"])
def test_execution_result_rejects_non_string_fields(field):
    with pytest.raises(ExecutionError, match=f"ExecutionResult.{field}"):
        make_result(**{field: 123})


@pytest.mark.parametrize("value", [None, "a state", {"user_task": "task"}, 42])
def test_execution_result_rejects_a_state_that_is_not_an_agent_state(value):
    with pytest.raises(ExecutionError, match="ExecutionResult.state must be an AgentState"):
        make_result(state=value)


def test_execution_error_is_a_runtime_error():
    assert issubclass(ExecutionError, RuntimeError)


def test_execution_module_has_no_llm_worker_or_router_dependency():
    from orchestrator_worker import execution as execution_module

    tree = ast.parse(pathlib.Path(execution_module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))

    assert imported == ["__future__", "dataclasses", ".state"]