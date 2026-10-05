"""Phase 5D tests: worker capabilities and capability-aware planning."""

from __future__ import annotations

import ast
import dataclasses
import json
import pathlib

import pytest

from fakes import FakeLLMClient, fake_response
from orchestrator_worker import application
from orchestrator_worker import capability as capability_module
from orchestrator_worker.capability import CapabilityError, WorkerCapability
from orchestrator_worker.plan import PlanStep
from orchestrator_worker.planner import LLMPlanner, UnknownWorkerError
from orchestrator_worker.router import Router
from orchestrator_worker.tool_loop import ToolLoop
from orchestrator_worker.tools import ToolExecutor, ToolRegistry
from orchestrator_worker.workers import (
    BaseWorker,
    CodingWorker,
    RegistryError,
    ResearchWorker,
    WorkerRegistry,
)
from orchestrator_worker.workers.research_tool import ResearchToolWorker

VALID_PLAN = {
    "goal": "a goal",
    "steps": [{"id": "step_1", "task": "a task", "worker_name": "research"}],
}


class MockWorker(BaseWorker):
    """Test worker whose capability is supplied at construction time."""

    def __init__(self, capability: WorkerCapability, output: str = "mock output") -> None:
        self._capability = capability
        self.name = capability.name
        self.output = output
        self.calls: list[str] = []

    @property
    def capability(self) -> WorkerCapability:
        return self._capability

    def execute(self, task: str) -> str:
        self.calls.append(task)
        return self.output


def build_registry() -> WorkerRegistry:
    registry = WorkerRegistry()
    registry.register(CodingWorker(FakeLLMClient()))
    registry.register(ResearchWorker(FakeLLMClient()))
    return registry


def research_tool_worker() -> ResearchToolWorker:
    loop = ToolLoop(FakeLLMClient(), ToolExecutor(ToolRegistry()))
    return ResearchToolWorker(loop, tools=())


def plan_json(worker_name: str, task: str = "a task") -> str:
    return json.dumps(
        {
            "goal": "a goal",
            "steps": [{"id": "step_1", "task": task, "worker_name": worker_name}],
        }
    )


# --------------------------------------------------- the capability value object


def test_capability_is_a_frozen_value_object():
    capability = WorkerCapability(name="research", description="does research")
    assert capability.name == "research"
    assert capability.description == "does research"
    assert capability == WorkerCapability("research", "does research")
    with pytest.raises(dataclasses.FrozenInstanceError):
        capability.name = "other"


@pytest.mark.parametrize("name", ["", "   ", None, 123])
def test_capability_rejects_an_invalid_name(name):
    with pytest.raises(CapabilityError):
        WorkerCapability(name=name, description="ok")


@pytest.mark.parametrize("description", ["", "   ", None, 123])
def test_capability_rejects_an_invalid_description(description):
    with pytest.raises(CapabilityError):
        WorkerCapability(name="research", description=description)


def test_capability_error_is_a_runtime_error():
    assert issubclass(CapabilityError, RuntimeError)


def test_capability_module_depends_only_on_the_standard_library():
    tree = ast.parse(pathlib.Path(capability_module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))
    assert imported == ["__future__", "dataclasses"]


def test_capability_module_is_provider_neutral():
    source = pathlib.Path(capability_module.__file__).read_text(encoding="utf-8")
    for forbidden in ("openai", "deepseek", "LLMClient", "ToolLoop", "ToolExecutor"):
        assert forbidden not in source


# --------------------------------------------- concrete workers expose capability


def test_research_worker_exposes_its_capability():
    capability = ResearchWorker(FakeLLMClient()).capability
    assert capability.name == "research"
    description = capability.description.lower()
    for term in ("research", "analysis", "information synthesis", "investigation"):
        assert term in description


def test_research_tool_worker_exposes_a_tool_aware_capability():
    capability = research_tool_worker().capability
    assert capability.name == "research"
    description = capability.description.lower()
    assert "research" in description
    assert "calculator" in description


def test_coding_worker_exposes_its_capability():
    capability = CodingWorker(FakeLLMClient()).capability
    assert capability.name == "coding"
    description = capability.description.lower()
    for term in ("software development", "implementation", "debugging", "analysis"):
        assert term in description


def test_base_worker_capability_defaults_to_a_general_description():
    class MinimalWorker(BaseWorker):
        name = "minimal"

        def execute(self, task: str) -> str:
            return task

    capability = MinimalWorker().capability
    assert capability.name == "minimal"
    assert capability.description.strip()


# ------------------------------------------- the registry is the capability source


def test_registry_capabilities_are_derived_from_the_registered_workers():
    registry = build_registry()
    capabilities = registry.capabilities()
    assert [capability.name for capability in capabilities] == ["coding", "research"]
    assert capabilities[0] == registry.get("coding").capability
    assert capabilities[1] == registry.get("research").capability


def test_registry_capabilities_are_empty_without_workers():
    assert WorkerRegistry().capabilities() == []


def test_registry_capabilities_track_registration_not_a_second_config():
    registry = WorkerRegistry()
    worker = MockWorker(WorkerCapability("mock_worker", "Handles mock tasks."))
    registry.register(worker)
    assert registry.capabilities() == [worker.capability]


# ------------------------------------------------- capability-aware planner input


def test_planner_prompt_includes_worker_capabilities():
    registry = build_registry()
    llm = FakeLLMClient(fake_response(json.dumps(VALID_PLAN)))
    planner = LLMPlanner(llm, available_workers=registry.capabilities())
    planner.create_plan("task")

    prompt = llm.requests[0].messages[0].content
    assert "- research" in prompt
    assert "- coding" in prompt
    assert registry.get("research").capability.description in prompt
    assert registry.get("coding").capability.description in prompt


def test_planner_still_accepts_plain_worker_names():
    llm = FakeLLMClient(fake_response(json.dumps(VALID_PLAN)))
    planner = LLMPlanner(llm, available_workers=["research", "coding"])
    planner.create_plan("task")
    prompt = llm.requests[0].messages[0].content
    assert "- research" in prompt
    assert planner.available_workers == ("coding", "research")


def test_planner_deduplicates_and_sorts_capabilities():
    capabilities = [
        WorkerCapability("research", "research capability"),
        WorkerCapability("coding", "coding capability"),
        WorkerCapability("research", "research capability"),
    ]
    llm = FakeLLMClient(fake_response(json.dumps(VALID_PLAN)))
    planner = LLMPlanner(llm, available_workers=capabilities)
    assert planner.available_workers == ("coding", "research")
    planner.create_plan("task")
    prompt = llm.requests[0].messages[0].content
    assert prompt.count("Capability: research capability") == 1


@pytest.mark.parametrize("names", [[""], ["research", "   "], ["research", None], [123]])
def test_planner_still_rejects_invalid_worker_entries(names):
    with pytest.raises(ValueError, match="non-empty strings"):
        LLMPlanner(FakeLLMClient(), available_workers=names)


# --------------------------------------------------- selection stays the model's


def test_planner_returns_the_models_research_choice():
    llm = FakeLLMClient(fake_response(plan_json("research")))
    planner = LLMPlanner(llm, available_workers=build_registry().capabilities())
    plan = planner.create_plan("please research something")
    assert plan.steps == [PlanStep(id="step_1", task="a task", worker_name="research")]


def test_planner_returns_the_models_coding_choice():
    llm = FakeLLMClient(fake_response(plan_json("coding")))
    planner = LLMPlanner(llm, available_workers=build_registry().capabilities())
    plan = planner.create_plan("please research something")
    assert plan.steps[0].worker_name == "coding"


def test_planner_does_not_route_by_task_keywords():
    # The task text points at research, but the model's plan chooses coding.
    # A deterministic planner must return the model's choice, not re-route it.
    llm = FakeLLMClient(fake_response(plan_json("coding", task="research and summarise")))
    planner = LLMPlanner(llm, available_workers=build_registry().capabilities())
    plan = planner.create_plan("research and summarise this report")
    assert plan.steps[0].worker_name == "coding"


def test_unknown_worker_still_raises_without_fallback():
    llm = FakeLLMClient(fake_response(plan_json("translator")))
    planner = LLMPlanner(llm, available_workers=build_registry().capabilities())
    with pytest.raises(UnknownWorkerError):
        planner.create_plan("task")
    assert len(llm.requests) == 1


# ---------------------------------------------- dynamic workers need no code change


def test_dynamic_worker_capability_reaches_the_planner():
    registry = build_registry()
    registry.register(
        MockWorker(WorkerCapability("mock_worker", "Handles mock investigations."))
    )

    llm = FakeLLMClient(fake_response(plan_json("mock_worker")))
    planner = LLMPlanner(llm, available_workers=registry.capabilities())
    plan = planner.create_plan("do a mock task")

    assert plan.steps[0].worker_name == "mock_worker"
    prompt = llm.requests[0].messages[0].content
    assert "- mock_worker" in prompt
    assert "Handles mock investigations." in prompt


# ------------------------------------------------------- composition root wiring


def test_application_planner_sees_registry_capabilities():
    llm = FakeLLMClient(fake_response(plan_json("research")))
    orchestrator = application.build_orchestrator(llm)
    orchestrator.plan("task")

    prompt = llm.requests[0].messages[0].content
    registry = orchestrator._registry
    assert "- research" in prompt
    assert "- coding" in prompt
    assert registry.get("research").capability.description in prompt
    assert registry.get("coding").capability.description in prompt
    assert [capability.name for capability in registry.capabilities()] == list(
        orchestrator._planner.available_workers
    )


def test_application_research_capability_mentions_the_calculator():
    orchestrator = application.build_orchestrator(FakeLLMClient())
    capability = orchestrator._registry.get("research").capability
    assert capability.name == "research"
    assert "calculator" in capability.description.lower()


# ----------------------------------------------------------- router stays deterministic


def test_router_stays_deterministic_and_has_no_fallback():
    registry = build_registry()
    router = Router(registry)
    assert router.resolve("research") is registry.get("research")
    with pytest.raises(RegistryError):
        router.resolve("unknown")