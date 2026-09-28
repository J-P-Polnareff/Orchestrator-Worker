"""Tests for WorkerRegistry."""

from __future__ import annotations

import ast
import pathlib

import pytest

from fakes import FakeLLMClient, NamedWorker, StubWorker
from orchestrator_worker.workers import CodingWorker, RegistryError, WorkerRegistry
from orchestrator_worker.workers import registry as registry_module


def test_register_then_get_returns_the_same_instance():
    registry = WorkerRegistry()
    worker = NamedWorker("research")
    registry.register(worker)
    assert registry.get("research") is worker


def test_registry_accepts_any_base_worker_subclass():
    registry = WorkerRegistry()
    registry.register(StubWorker())
    registry.register(NamedWorker("summarise"))
    registry.register(CodingWorker(FakeLLMClient()))
    assert registry.names() == ["coding", "stub", "summarise"]


def test_has_reports_registration_state():
    registry = WorkerRegistry()
    assert registry.has("research") is False
    registry.register(NamedWorker("research"))
    assert registry.has("research") is True


def test_names_are_sorted():
    registry = WorkerRegistry()
    registry.register(NamedWorker("coding"))
    registry.register(NamedWorker("research"))
    assert registry.names() == ["coding", "research"]


def test_get_unknown_worker_raises_with_available_names():
    registry = WorkerRegistry()
    registry.register(NamedWorker("research"))
    registry.register(NamedWorker("coding"))

    with pytest.raises(RegistryError) as excinfo:
        registry.get("summarise")

    message = str(excinfo.value)
    assert "summarise" in message
    assert "coding" in message and "research" in message


def test_get_on_empty_registry_says_none_are_registered():
    with pytest.raises(RegistryError, match=r"\(none\)"):
        WorkerRegistry().get("research")


def test_unknown_lookup_does_not_fall_back_to_a_registered_worker():
    registry = WorkerRegistry()
    research = NamedWorker("research")
    registry.register(research)

    with pytest.raises(RegistryError):
        registry.get("coding")

    assert research.calls == []


def test_duplicate_registration_is_rejected():
    registry = WorkerRegistry()
    registry.register(NamedWorker("research", output="first"))

    with pytest.raises(RegistryError, match="already registered"):
        registry.register(NamedWorker("research", output="second"))


def test_duplicate_registration_keeps_the_original_worker():
    registry = WorkerRegistry()
    first = NamedWorker("research", output="first")
    registry.register(first)

    with pytest.raises(RegistryError):
        registry.register(NamedWorker("research", output="second"))

    assert registry.get("research") is first


def test_registry_error_is_a_runtime_error():
    assert issubclass(RegistryError, RuntimeError)


def test_registry_only_depends_on_the_base_contract():
    tree = ast.parse(pathlib.Path(registry_module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))

    assert imported == ["__future__", ".base"]