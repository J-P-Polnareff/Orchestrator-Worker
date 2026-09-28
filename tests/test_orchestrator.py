"""Tests for the Phase 1 orchestrator."""

from __future__ import annotations

import pytest

from fakes import FakeLLMClient, StubWorker, fake_response
from orchestrator_worker.orchestrator import Orchestrator, OrchestratorError
from orchestrator_worker.state import AgentState
from orchestrator_worker.workers import ResearchWorker, WorkerError


def test_run_returns_a_completed_state():
    worker = StubWorker(output="the answer")
    state = Orchestrator(worker).run("a question")

    assert isinstance(state, AgentState)
    assert state.user_task == "a question"
    assert state.worker_name == "stub"
    assert state.worker_input == "a question"
    assert state.worker_output == "the answer"
    assert state.final_result == "the answer"


def test_worker_receives_the_user_task():
    worker = StubWorker()
    Orchestrator(worker).run("a question")
    assert worker.calls == ["a question"]


def test_user_task_is_stripped_before_routing():
    worker = StubWorker()
    state = Orchestrator(worker).run("  a question  ")
    assert state.user_task == "a question"
    assert worker.calls == ["a question"]


def test_routing_follows_the_injected_worker():
    class OtherWorker(StubWorker):
        name = "other"

    state = Orchestrator(OtherWorker()).run("task")
    assert state.worker_name == "other"


def test_worker_name_property_reports_the_target():
    assert Orchestrator(StubWorker()).worker_name == "stub"


@pytest.mark.parametrize("task", ["", "   ", "\n"])
def test_empty_user_task_is_rejected_before_the_worker_runs(task):
    worker = StubWorker()
    with pytest.raises(OrchestratorError, match="non-empty"):
        Orchestrator(worker).run(task)
    assert worker.calls == []


def test_worker_failure_is_wrapped_and_chained():
    worker = StubWorker(error=WorkerError("research worker LLM call failed: boom"))
    orchestrator = Orchestrator(worker)

    with pytest.raises(OrchestratorError) as excinfo:
        orchestrator.run("task")

    assert "stub" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, WorkerError)


def test_unexpected_worker_errors_are_not_swallowed():
    worker = StubWorker(error=ValueError("unexpected"))
    with pytest.raises(ValueError):
        Orchestrator(worker).run("task")


def test_end_to_end_with_research_worker_and_fake_llm():
    llm = FakeLLMClient(fake_response("DeepSeek V3 uses a Mixture-of-Experts architecture."))
    orchestrator = Orchestrator(ResearchWorker(llm))

    state = orchestrator.run("What architecture does DeepSeek V3 use?")

    assert state.worker_name == "research"
    assert state.worker_input == "What architecture does DeepSeek V3 use?"
    assert state.final_result == "DeepSeek V3 uses a Mixture-of-Experts architecture."
    assert state.worker_output == state.final_result
    assert len(llm.requests) == 1


def test_end_to_end_llm_failure_surfaces_as_orchestrator_error():
    from orchestrator_worker.llm import LLMError

    llm = FakeLLMClient(error=LLMError("connection reset"))
    with pytest.raises(OrchestratorError) as excinfo:
        Orchestrator(ResearchWorker(llm)).run("task")
    assert isinstance(excinfo.value.__cause__, WorkerError)


def test_result_is_serialisable():
    state = Orchestrator(StubWorker(output="answer")).run("task")
    assert state.to_dict() == {
        "user_task": "task",
        "worker_name": "stub",
        "worker_input": "task",
        "worker_output": "answer",
        "final_result": "answer",
    }