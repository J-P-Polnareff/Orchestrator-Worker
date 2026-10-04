"""Phase 5A tests: the application composition root."""

from __future__ import annotations

import ast
import pathlib

from fakes import FakeLLMClient, SequenceLLMClient, fake_response
from orchestrator_worker import application, config
from orchestrator_worker.aggregators import LLMAggregator
from orchestrator_worker.application import build_orchestrator
from orchestrator_worker.evaluators import LLMEvaluator
from orchestrator_worker.llm import LLMRequest
from orchestrator_worker.orchestrator import Orchestrator
from orchestrator_worker.planner import LLMPlanner
from orchestrator_worker.workers import CodingWorker, ResearchWorker, WorkerRegistry


def test_build_orchestrator_returns_a_configured_orchestrator():
    orchestrator = build_orchestrator(FakeLLMClient())

    assert isinstance(orchestrator, Orchestrator)
    assert isinstance(orchestrator._registry, WorkerRegistry)
    assert isinstance(orchestrator._planner, LLMPlanner)
    assert isinstance(orchestrator._evaluator, LLMEvaluator)
    assert isinstance(orchestrator._aggregator, LLMAggregator)


def test_registry_contains_exactly_the_research_and_coding_workers():
    orchestrator = build_orchestrator(FakeLLMClient())
    registry = orchestrator._registry

    assert registry.names() == ["coding", "research"]
    assert isinstance(registry.get("research"), ResearchWorker)
    assert isinstance(registry.get("coding"), CodingWorker)


def test_planner_available_workers_match_the_registry():
    orchestrator = build_orchestrator(FakeLLMClient())
    registry = orchestrator._registry

    assert sorted(orchestrator._planner.available_workers) == registry.names()


def test_every_llm_backed_component_shares_the_injected_client():
    llm = FakeLLMClient()
    orchestrator = build_orchestrator(llm)
    registry = orchestrator._registry

    assert orchestrator._planner._llm is llm
    assert orchestrator._evaluator._llm is llm
    assert orchestrator._aggregator._llm is llm
    assert registry.get("research")._llm is llm
    assert registry.get("coding")._llm is llm


def test_building_does_not_call_the_model():
    llm = FakeLLMClient()

    build_orchestrator(llm)

    assert llm.requests == []


def test_injected_client_never_builds_a_real_provider_client(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("build_llm_client must not be called")

    monkeypatch.setattr(application, "build_llm_client", boom)

    assert isinstance(build_orchestrator(FakeLLMClient()), Orchestrator)


def test_injected_client_needs_no_env_or_api_key(monkeypatch):
    def boom(cls, *args, **kwargs):
        raise AssertionError("Settings.from_env must not be called")

    monkeypatch.setattr(config.Settings, "from_env", classmethod(boom))

    assert isinstance(build_orchestrator(FakeLLMClient()), Orchestrator)


def test_provider_factory_is_used_only_when_no_client_is_given(monkeypatch):
    calls: list[bool] = []
    llm = FakeLLMClient()

    def factory():
        calls.append(True)
        return llm

    monkeypatch.setattr(application, "build_llm_client", factory)

    orchestrator = build_orchestrator()

    assert calls == [True]
    assert orchestrator._planner._llm is llm


PLANNER_JSON = (
    '{"goal": "research asyncio and write an example", "steps": ['
    '{"id": "step_1", "task": "study asyncio", "worker_name": "research"}, '
    '{"id": "step_2", "task": "write an example", "worker_name": "coding"}]}'
)


def build_pipeline_client() -> SequenceLLMClient:
    return SequenceLLMClient(
        [
            fake_response(PLANNER_JSON),
            fake_response("asyncio is a library for concurrent code."),
            fake_response("import asyncio\n\nasync def main(): ..."),
            fake_response('{"passed": true, "reason": "results look complete"}'),
            fake_response("FINAL ANSWER"),
        ]
    )


def stage_of(request: LLMRequest) -> str:
    system = request.messages[0].content
    prefixes = {
        "You are a planning component.": "planner",
        "You are a research worker.": "research",
        "You are a coding worker.": "coding",
        "You are an execution-result evaluator.": "evaluator",
        "You are an aggregation component.": "aggregator",
    }
    for prefix, stage in prefixes.items():
        if system.startswith(prefix):
            return stage
    raise AssertionError(f"unrecognised pipeline request: {system[:60]!r}")


def test_composed_orchestrator_runs_the_full_pipeline():
    llm = build_pipeline_client()
    orchestrator = build_orchestrator(llm)

    answer = orchestrator.run_pipeline("research asyncio and write an example")

    assert answer == "FINAL ANSWER"
    assert [stage_of(request) for request in llm.requests] == [
        "planner",
        "research",
        "coding",
        "evaluator",
        "aggregator",
    ]


def test_composed_pipeline_produces_a_result_per_plan_step():
    llm = build_pipeline_client()
    orchestrator = build_orchestrator(llm)

    plan = orchestrator.plan("research asyncio and write an example")
    context = orchestrator.execute_plan_context(plan)

    assert plan.goal == "research asyncio and write an example"
    assert [result.step_id for result in context.results] == ["step_1", "step_2"]
    assert [result.worker_name for result in context.results] == [
        "research",
        "coding",
    ]


def test_orchestrator_module_does_not_import_the_composition_root():
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

    assert not any("application" in name for name in imported)


def test_application_module_has_no_dependency_cycle():
    from orchestrator_worker import application as application_module

    tree = ast.parse(
        pathlib.Path(application_module.__file__).read_text(encoding="utf-8")
    )
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))

    # The composition root may use concrete components, but it must never be
    # imported back by them.
    assert ".orchestrator" in imported
    assert not any(name.endswith(".application") for name in imported)