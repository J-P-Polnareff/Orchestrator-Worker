"""Phase 2 tests: the orchestrator routes through the worker registry."""

from __future__ import annotations

import ast
import pathlib

import pytest

from fakes import FakeLLMClient, NamedWorker, StubWorker, fake_response
from orchestrator_worker.llm import LLMError
from orchestrator_worker.orchestrator import (
    DEFAULT_WORKER_NAME,
    Orchestrator,
    OrchestratorError,
)
from orchestrator_worker.workers import (
    CodingWorker,
    RegistryError,
    ResearchWorker,
    WorkerError,
    WorkerRegistry,
)


def build_registry() -> WorkerRegistry:
    """Registry holding a research and a coding worker, both backed by fakes."""
    registry = WorkerRegistry()
    registry.register(ResearchWorker(FakeLLMClient(fake_response("research answer"))))
    registry.register(CodingWorker(FakeLLMClient(fake_response("coding answer"))))
    return registry


def test_default_worker_is_research():
    assert DEFAULT_WORKER_NAME == "research"
    assert Orchestrator(build_registry()).worker_name == "research"


def test_no_worker_name_still_routes_to_research():
    state = Orchestrator(build_registry()).run("a question")
    assert state.worker_name == "research"
    assert state.final_result == "research answer"


def test_explicit_research_worker_is_selected():
    state = Orchestrator(build_registry()).run("a question", worker_name="research")
    assert state.worker_name == "research"
    assert state.final_result == "research answer"


def test_explicit_coding_worker_is_selected():
    state = Orchestrator(build_registry()).run("refactor this", worker_name="coding")
    assert state.worker_name == "coding"
    assert state.final_result == "coding answer"


def test_explicit_worker_name_overrides_the_default():
    orchestrator = Orchestrator(build_registry())
    assert orchestrator.run("task", worker_name="coding").worker_name == "coding"
    assert orchestrator.run("task").worker_name == "research"


def test_worker_input_is_recorded_for_the_selected_worker():
    state = Orchestrator(build_registry()).run("  refactor this  ", worker_name="coding")
    assert state.user_task == "refactor this"
    assert state.worker_input == "refactor this"
    assert state.worker_output == state.final_result


def test_unknown_worker_raises_a_clear_registry_error():
    with pytest.raises(RegistryError) as excinfo:
        Orchestrator(build_registry()).run("task", worker_name="translate")

    message = str(excinfo.value)
    assert "translate" in message
    assert "coding" in message and "research" in message


def test_unknown_worker_does_not_fall_back_to_research():
    research = NamedWorker("research")
    registry = WorkerRegistry()
    registry.register(research)
    registry.register(CodingWorker(FakeLLMClient(fake_response("coding answer"))))

    with pytest.raises(RegistryError):
        Orchestrator(registry).run("task", worker_name="translate")

    assert research.calls == []


def test_empty_task_is_still_rejected_before_lookup():
    with pytest.raises(OrchestratorError, match="non-empty"):
        Orchestrator(build_registry()).run("   ", worker_name="coding")


def test_worker_failure_in_coding_worker_is_wrapped_consistently():
    registry = WorkerRegistry()
    registry.register(ResearchWorker(FakeLLMClient(fake_response("ok"))))
    registry.register(CodingWorker(FakeLLMClient(error=LLMError("boom"))))

    with pytest.raises(OrchestratorError) as excinfo:
        Orchestrator(registry).run("task", worker_name="coding")

    assert "coding" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, WorkerError)


def test_phase1_single_worker_shorthand_still_works():
    state = Orchestrator(StubWorker(output="stub answer")).run("task")
    assert state.worker_name == "stub"
    assert state.final_result == "stub answer"


def test_orchestrator_does_not_import_concrete_workers():
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
        ".aggregators.base",
        ".context",
        ".execution",
        ".plan",
        ".planner.base",
        ".router",
        ".state",
        ".workers.base",
        ".workers.registry",
    ]
    assert not any("research" in name or "coding" in name for name in imported)


def test_orchestrator_reads_the_registry_at_call_time():
    registry = build_registry()
    orchestrator = Orchestrator(registry)

    registry.register(NamedWorker("extra", output="extra output"))

    state = orchestrator.run("task", worker_name="extra")
    assert state.final_result == "extra output"