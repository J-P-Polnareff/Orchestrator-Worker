"""Phase 4A tests: the router resolves worker names deterministically."""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

from fakes import NamedWorker, SpyRegistry
from orchestrator_worker.router import Router
from orchestrator_worker.workers import RegistryError, WorkerRegistry


def build_registry() -> tuple[WorkerRegistry, NamedWorker, NamedWorker]:
    research = NamedWorker("research", output="research answer")
    coding = NamedWorker("coding", output="coding answer")
    registry = WorkerRegistry()
    registry.register(research)
    registry.register(coding)
    return registry, research, coding


def test_resolve_returns_the_research_worker():
    registry, research, _ = build_registry()
    resolved = Router(registry).resolve("research")
    assert resolved is research


def test_resolve_returns_the_coding_worker():
    registry, _, coding = build_registry()
    resolved = Router(registry).resolve("coding")
    assert resolved is coding


def test_resolve_uses_the_registry_lookup_for_the_requested_name():
    registry = SpyRegistry()
    registry.register(NamedWorker("research"))
    Router(registry).resolve("research")
    assert registry.get_calls == ["research"]


def test_unknown_worker_raises_a_clear_registry_error():
    registry, _, _ = build_registry()
    with pytest.raises(RegistryError) as excinfo:
        Router(registry).resolve("translate")

    message = str(excinfo.value)
    assert "translate" in message
    assert "coding" in message and "research" in message


def test_unknown_worker_does_not_fall_back_to_a_registered_worker():
    registry, research, _ = build_registry()
    with pytest.raises(RegistryError):
        Router(registry).resolve("reserch")
    assert research.calls == []


def test_resolve_only_accepts_a_worker_name():
    assert list(inspect.signature(Router.resolve).parameters) == ["self", "worker_name"]


def test_router_reads_the_registry_at_call_time():
    registry = WorkerRegistry()
    router = Router(registry)

    late = NamedWorker("late", output="late answer")
    registry.register(late)

    assert router.resolve("late") is late


def test_router_has_no_llm_dependency():
    registry, _, _ = build_registry()
    router = Router(registry)
    assert not hasattr(router, "_llm")
    assert not hasattr(router, "complete")


def test_router_module_does_not_import_llm_or_concrete_workers():
    from orchestrator_worker import router as router_module

    tree = ast.parse(pathlib.Path(router_module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))

    assert imported == ["__future__", ".workers.base", ".workers.registry"]
    assert not any("llm" in name or "deepseek" in name for name in imported)
    assert not any("research" in name or "coding" in name for name in imported)