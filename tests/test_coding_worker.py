"""Tests for CodingWorker."""

from __future__ import annotations

import ast
import pathlib

import pytest

from fakes import FakeLLMClient, fake_response
from orchestrator_worker.llm import LLMClient, LLMError, LLMRequest, LLMResponse, Message
from orchestrator_worker.workers import BaseWorker, CodingWorker, WorkerError
from orchestrator_worker.workers import coding as coding_module
from orchestrator_worker.workers import research as research_module


def test_coding_worker_identity():
    worker = CodingWorker(FakeLLMClient())
    assert worker.name == "coding"
    assert isinstance(worker, BaseWorker)


def test_execute_returns_the_llm_content():
    llm = FakeLLMClient(fake_response("Use a dict comprehension."))
    assert CodingWorker(llm).execute("How do I build a dict?") == "Use a dict comprehension."


def test_execute_strips_surrounding_whitespace():
    llm = FakeLLMClient(fake_response("  padded answer  "))
    assert CodingWorker(llm).execute("task") == "padded answer"


def test_execute_sends_a_coding_system_prompt_and_the_task():
    llm = FakeLLMClient(fake_response("ok"))
    CodingWorker(llm).execute("Refactor this function")

    request = llm.requests[0]
    assert request.messages[0].role == "system"
    assert request.messages[0].content == coding_module.DEFAULT_SYSTEM_PROMPT
    assert request.messages[1] == Message.user("Refactor this function")


def test_default_system_prompt_is_coding_oriented():
    prompt = coding_module.DEFAULT_SYSTEM_PROMPT.lower()
    assert "coding" in prompt
    assert prompt != research_module.DEFAULT_SYSTEM_PROMPT.lower()


def test_custom_system_prompt_is_used():
    llm = FakeLLMClient(fake_response("ok"))
    CodingWorker(llm, system_prompt="be terse").execute("task")
    assert llm.requests[0].messages[0] == Message.system("be terse")


def test_optional_parameters_are_forwarded_when_set():
    llm = FakeLLMClient(fake_response("ok"))
    CodingWorker(llm, temperature=0.1, max_tokens=64).execute("task")

    request = llm.requests[0]
    assert request.temperature == 0.1
    assert request.max_tokens == 64


def test_optional_parameters_default_to_none():
    llm = FakeLLMClient(fake_response("ok"))
    CodingWorker(llm).execute("task")

    request = llm.requests[0]
    assert request.temperature is None
    assert request.max_tokens is None


@pytest.mark.parametrize("task", ["", "   ", "\n\t"])
def test_empty_task_raises_without_calling_the_llm(task):
    llm = FakeLLMClient(fake_response("should not be used"))
    with pytest.raises(WorkerError, match="empty task"):
        CodingWorker(llm).execute(task)
    assert llm.requests == []


def test_llm_error_is_wrapped_as_worker_error():
    llm = FakeLLMClient(error=LLMError("DeepSeek chat completion failed: boom"))
    with pytest.raises(WorkerError, match="coding worker LLM call failed") as excinfo:
        CodingWorker(llm).execute("task")
    assert isinstance(excinfo.value.__cause__, LLMError)
    assert llm.requests, "the worker should have attempted the call"


def test_empty_llm_response_is_rejected():
    llm = FakeLLMClient(fake_response("   "))
    with pytest.raises(WorkerError, match="empty response"):
        CodingWorker(llm).execute("task")


def test_coding_worker_only_depends_on_the_llm_abstraction():
    tree = ast.parse(pathlib.Path(coding_module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))

    assert imported == ["__future__", "..llm", ".base"]
    assert not any("deepseek" in name or "openai" in name for name in imported)


def test_coding_worker_accepts_any_llm_client():
    class MinimalLLM(LLMClient):
        name = "minimal"

        def complete(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(content="from minimal client", model="minimal")

    assert CodingWorker(MinimalLLM()).execute("task") == "from minimal client"


def test_coding_worker_does_not_execute_code():
    source = pathlib.Path(coding_module.__file__).read_text(encoding="utf-8")
    for forbidden in ("subprocess", "os.system", "eval(", "exec("):
        assert forbidden not in source