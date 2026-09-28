"""Tests for the worker contract and ResearchWorker."""

from __future__ import annotations

import ast
import pathlib

import pytest

from fakes import FakeLLMClient, fake_response
from orchestrator_worker.llm import LLMClient, LLMError, LLMRequest, LLMResponse, Message
from orchestrator_worker.workers import BaseWorker, ResearchWorker, WorkerError
from orchestrator_worker.workers import research as research_module


class CompleteWorker(BaseWorker):
    name = "complete"

    def execute(self, task: str) -> str:
        return f"handled: {task}"


def test_base_worker_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseWorker()


def test_subclass_without_execute_cannot_be_instantiated():
    class Incomplete(BaseWorker):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()


def test_subclass_with_execute_satisfies_the_contract():
    worker = CompleteWorker()
    assert isinstance(worker, BaseWorker)
    assert worker.name == "complete"
    assert worker.execute("x") == "handled: x"


def test_worker_error_is_a_runtime_error():
    assert issubclass(WorkerError, RuntimeError)


def test_research_worker_identity():
    worker = ResearchWorker(FakeLLMClient())
    assert worker.name == "research"
    assert isinstance(worker, BaseWorker)


def test_execute_sends_system_and_user_messages():
    llm = FakeLLMClient(fake_response("DeepSeek V3 is a MoE model."))
    worker = ResearchWorker(llm)

    result = worker.execute("What is DeepSeek V3?")

    assert result == "DeepSeek V3 is a MoE model."
    assert len(llm.requests) == 1
    request = llm.requests[0]
    assert request.messages[0].role == "system"
    assert request.messages[1] == Message.user("What is DeepSeek V3?")


def test_execute_strips_surrounding_whitespace():
    llm = FakeLLMClient(fake_response("  padded answer  "))
    assert ResearchWorker(llm).execute("task") == "padded answer"


def test_optional_parameters_are_forwarded_when_set():
    llm = FakeLLMClient(fake_response("ok"))
    ResearchWorker(llm, temperature=0.2, max_tokens=256).execute("task")

    request = llm.requests[0]
    assert request.temperature == 0.2
    assert request.max_tokens == 256


def test_optional_parameters_default_to_none():
    llm = FakeLLMClient(fake_response("ok"))
    ResearchWorker(llm).execute("task")

    request = llm.requests[0]
    assert request.temperature is None
    assert request.max_tokens is None


def test_custom_system_prompt_is_used():
    llm = FakeLLMClient(fake_response("ok"))
    ResearchWorker(llm, system_prompt="be terse").execute("task")
    assert llm.requests[0].messages[0] == Message.system("be terse")


@pytest.mark.parametrize("task", ["", "   ", "\n\t"])
def test_empty_task_raises_without_calling_the_llm(task):
    llm = FakeLLMClient(fake_response("should not be used"))
    with pytest.raises(WorkerError, match="empty task"):
        ResearchWorker(llm).execute(task)
    assert llm.requests == []


def test_llm_error_is_wrapped_as_worker_error():
    llm = FakeLLMClient(error=LLMError("DeepSeek chat completion failed: boom"))
    with pytest.raises(WorkerError, match="LLM call failed"):
        ResearchWorker(llm).execute("task")
    assert llm.requests, "the worker should have attempted the call"


def test_empty_llm_response_is_rejected():
    llm = FakeLLMClient(fake_response("   "))
    with pytest.raises(WorkerError, match="empty response"):
        ResearchWorker(llm).execute("task")


def test_research_worker_only_depends_on_the_llm_abstraction():
    tree = ast.parse(pathlib.Path(research_module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))

    assert imported == ["__future__", "..llm", ".base"]
    assert not any("deepseek" in name or "openai" in name for name in imported)



def test_research_worker_accepts_any_llm_client():
    class MinimalLLM(LLMClient):
        name = "minimal"

        def complete(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(content="from minimal client", model="minimal")

    assert ResearchWorker(MinimalLLM()).execute("task") == "from minimal client"