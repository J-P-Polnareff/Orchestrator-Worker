"""Tests for AgentState."""

from __future__ import annotations

from orchestrator_worker.state import AgentState


def test_only_user_task_is_required():
    state = AgentState(user_task="Explain transformers")
    assert state.user_task == "Explain transformers"
    assert state.worker_name is None
    assert state.worker_input is None
    assert state.worker_output is None
    assert state.final_result is None


def test_fields_are_mutable():
    state = AgentState(user_task="task")
    state.worker_name = "research"
    state.worker_input = "task"
    state.worker_output = "output"
    state.final_result = "output"
    assert state.worker_name == "research"
    assert state.final_result == "output"


def test_to_dict_exposes_every_field():
    state = AgentState(
        user_task="task",
        worker_name="research",
        worker_input="input",
        worker_output="output",
        final_result="final",
    )
    assert state.to_dict() == {
        "user_task": "task",
        "worker_name": "research",
        "worker_input": "input",
        "worker_output": "output",
        "final_result": "final",
    }


def test_instances_are_independent():
    first = AgentState(user_task="one")
    second = AgentState(user_task="two")
    first.final_result = "done"
    assert second.final_result is None