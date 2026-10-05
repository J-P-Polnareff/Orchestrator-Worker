"""Phase 5A tests: the application composition root."""

from __future__ import annotations

import ast
import pathlib

from fakes import FakeLLMClient, SequenceLLMClient, fake_response
from orchestrator_worker import application, config
from orchestrator_worker.aggregators import LLMAggregator
from orchestrator_worker.application import build_orchestrator
from orchestrator_worker.evaluators import LLMEvaluator
from orchestrator_worker.llm import LLMRequest, LLMResponse
from orchestrator_worker.tool_loop import ToolLoop
from orchestrator_worker.tools import Tool, ToolCall
from orchestrator_worker.orchestrator import Orchestrator
from orchestrator_worker.planner import LLMPlanner
from orchestrator_worker.workers import CodingWorker, WorkerRegistry
from orchestrator_worker.workers.research_tool import ResearchToolWorker


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
    assert isinstance(registry.get("research"), ResearchToolWorker)
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
    # The research worker is tool-enabled, so the client reaches it through
    # the injected ToolLoop instead of directly.
    assert registry.get("research")._tool_loop._llm is llm
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
            fake_response(
                '{"passed": true, "reason": "results look complete", '
                '"step_evaluations": ['
                '{"step_id": "step_1", "passed": true, "feedback": ""}, '
                '{"step_id": "step_2", "passed": true, "feedback": ""}]}'
            ),
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

# ------------------------------------------------- Phase 5C-4B tool wiring

RESEARCH_TOOL_PLAN_JSON = (
    '{"goal": "compute 17 x 23", "steps": ['
    '{"id": "research-1", "task": "compute 17 x 23 and explain it", '
    '"worker_name": "research"}]}'
)


def build_tool_pipeline_client() -> SequenceLLMClient:
    return SequenceLLMClient(
        [
            fake_response(RESEARCH_TOOL_PLAN_JSON),
            LLMResponse(
                content="",
                model="fake-model",
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="calculator",
                        arguments={"a": 17, "b": 23, "operation": "multiply"},
                    ),
                ),
            ),
            fake_response("17 x 23 = 391."),
            fake_response(
                '{"passed": true, "reason": "the calculator result supports it", '
                '"step_evaluations": ['
                '{"step_id": "research-1", "passed": true, "feedback": ""}]}'
            ),
            fake_response("17 x 23 = 391."),
        ]
    )


def build_tool_orchestrator(monkeypatch):
    """Build the real composition root with a spying calculator tool."""
    seen: list[dict] = []
    real = application.calculator_tool()

    def spy(arguments):
        seen.append(dict(arguments))
        return real.handler(arguments)

    spy_tool = Tool(
        name=real.name,
        description=real.description,
        parameters=real.parameters,
        handler=spy,
    )
    monkeypatch.setattr(application, "calculator_tool", lambda: spy_tool)

    llm = build_tool_pipeline_client()
    return build_orchestrator(llm), llm, seen


def test_registry_research_worker_is_tool_enabled():
    orchestrator = build_orchestrator(FakeLLMClient())
    worker = orchestrator._registry.get("research")

    assert isinstance(worker, ResearchToolWorker)
    assert worker.name == "research"
    assert [tool.name for tool in worker._tools] == ["calculator"]


def test_planner_lists_only_registry_worker_names():
    orchestrator = build_orchestrator(FakeLLMClient())

    assert orchestrator._planner.available_workers == ("coding", "research")
    assert "research_tool" not in orchestrator._planner.available_workers


def test_application_wires_the_tool_runtime_for_the_research_worker():
    orchestrator = build_orchestrator(FakeLLMClient())
    worker = orchestrator._registry.get("research")
    tool_loop = worker._tool_loop
    executor = tool_loop._executor

    assert isinstance(tool_loop, ToolLoop)
    assert executor._registry.names() == ["calculator"]
    assert worker._tools[0] is executor._registry.get("calculator")


def test_coding_worker_stays_tool_free():
    orchestrator = build_orchestrator(FakeLLMClient())
    coding = orchestrator._registry.get("coding")

    assert isinstance(coding, CodingWorker)
    assert not hasattr(coding, "_tool_loop")


def test_composed_pipeline_executes_the_calculator_tool(monkeypatch):
    orchestrator, llm, seen = build_tool_orchestrator(monkeypatch)

    answer = orchestrator.run_pipeline("compute 17 x 23")

    assert answer == "17 x 23 = 391."
    assert len(llm.requests) == 5
    assert [stage_of(request) for request in llm.requests] == [
        "planner",
        "research",
        "research",
        "evaluator",
        "aggregator",
    ]
    assert seen == [{"a": 17, "b": 23, "operation": "multiply"}]


def test_composed_pipeline_feeds_the_tool_result_back_to_the_model(monkeypatch):
    orchestrator, llm, _seen = build_tool_orchestrator(monkeypatch)

    orchestrator.run_pipeline("compute 17 x 23")

    follow_up = llm.requests[2]
    assert [message.role for message in follow_up.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert follow_up.messages[2].tool_calls[0].name == "calculator"
    assert follow_up.messages[2].tool_calls[0].arguments == {
        "a": 17,
        "b": 23,
        "operation": "multiply",
    }
    assert follow_up.messages[3].tool_call_id == "call-1"
    assert follow_up.messages[3].content == "391"


def test_composed_research_step_returns_the_worker_output(monkeypatch):
    orchestrator, _llm, _seen = build_tool_orchestrator(monkeypatch)

    plan = orchestrator.plan("compute 17 x 23")
    context = orchestrator.execute_plan_context(plan)

    assert [result.step_id for result in context.results] == ["research-1"]
    assert context.results[0].worker_name == "research"
    assert context.results[0].state.worker_output == "17 x 23 = 391."