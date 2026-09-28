"""Phase 4A tests: sequential plan execution through the router."""

from __future__ import annotations

import pytest

from fakes import (
    FakeLLMClient,
    NamedWorker,
    RecordingWorker,
    SpyPlanner,
    SpyRegistry,
    StubWorker,
    fake_response,
)
from orchestrator_worker.orchestrator import Orchestrator, OrchestratorError
from orchestrator_worker.plan import Plan, PlanStep
from orchestrator_worker.planner import LLMPlanner
from orchestrator_worker.workers import (
    BaseWorker,
    RegistryError,
    WorkerError,
    WorkerRegistry,
)


def step(step_id: str, task: str, worker_name: str) -> PlanStep:
    return PlanStep(id=step_id, task=task, worker_name=worker_name)


def build_plan(*steps: PlanStep) -> Plan:
    return Plan(goal="a goal", steps=list(steps))


def build_registry(*workers: BaseWorker) -> WorkerRegistry:
    registry = WorkerRegistry()
    for worker in workers:
        registry.register(worker)
    return registry


def test_execute_plan_runs_a_single_step():
    research = NamedWorker("research", output="research answer")
    orchestrator = Orchestrator(build_registry(research))

    states = orchestrator.execute_plan(build_plan(step("step_1", "study asyncio", "research")))

    assert len(states) == 1
    assert states[0].worker_name == "research"
    assert states[0].worker_output == "research answer"
    assert states[0].final_result == "research answer"
    assert research.calls == ["study asyncio"]


def test_each_state_records_the_step_input_and_output():
    research = NamedWorker("research", output="research answer")
    orchestrator = Orchestrator(build_registry(research))

    state = orchestrator.execute_plan(
        build_plan(step("step_1", "study asyncio", "research"))
    )[0]

    assert state.user_task == "study asyncio"
    assert state.worker_input == "study asyncio"
    assert state.worker_output == "research answer"


def test_execute_plan_runs_steps_in_plan_order():
    log: list[str] = []
    research = RecordingWorker("research", log)
    coding = RecordingWorker("coding", log)
    orchestrator = Orchestrator(build_registry(research, coding))
    plan = build_plan(
        step("step_1", "study asyncio", "research"),
        step("step_2", "write an example", "coding"),
    )

    states = orchestrator.execute_plan(plan)

    assert log == ["research", "coding"]
    assert research.calls == ["study asyncio"]
    assert coding.calls == ["write an example"]
    assert [state.worker_name for state in states] == ["research", "coding"]
    assert len(states) == len(plan.steps)


def test_execute_plan_follows_plan_order_not_registry_order():
    log: list[str] = []
    research = RecordingWorker("research", log)
    coding = RecordingWorker("coding", log)
    orchestrator = Orchestrator(build_registry(research, coding))
    plan = build_plan(
        step("step_1", "write an example", "coding"),
        step("step_2", "study asyncio", "research"),
    )

    orchestrator.execute_plan(plan)

    assert log == ["coding", "research"]


def test_each_step_is_a_single_worker_call():
    log: list[str] = []
    research = RecordingWorker("research", log)
    coding = RecordingWorker("coding", log)
    orchestrator = Orchestrator(build_registry(research, coding))
    plan = build_plan(
        step("step_1", "study asyncio", "research"),
        step("step_2", "write an example", "coding"),
    )

    orchestrator.execute_plan(plan)

    assert len(research.calls) == 1
    assert len(coding.calls) == 1


def test_execute_plan_routes_each_step_through_the_registry():
    log: list[str] = []
    registry = SpyRegistry()
    registry.register(RecordingWorker("research", log))
    registry.register(RecordingWorker("coding", log))
    orchestrator = Orchestrator(registry)
    plan = build_plan(
        step("step_1", "study asyncio", "research"),
        step("step_2", "write an example", "coding"),
    )

    orchestrator.execute_plan(plan)

    assert registry.get_calls == ["research", "coding"]


def test_execute_plan_works_without_a_planner():
    research = NamedWorker("research", output="research answer")
    orchestrator = Orchestrator(build_registry(research))

    states = orchestrator.execute_plan(build_plan(step("step_1", "task", "research")))

    assert states[0].worker_output == "research answer"


def test_execute_plan_does_not_call_the_planner():
    planner = SpyPlanner()
    research = NamedWorker("research")
    orchestrator = Orchestrator(build_registry(research), planner=planner)

    orchestrator.execute_plan(build_plan(step("step_1", "task", "research")))

    assert planner.calls == []


def test_execute_plan_never_asks_an_llm_to_choose_a_worker():
    llm = FakeLLMClient(fake_response("unused"))
    planner = LLMPlanner(llm, available_workers=["research"])
    research = NamedWorker("research")
    orchestrator = Orchestrator(build_registry(research), planner=planner)

    orchestrator.execute_plan(build_plan(step("step_1", "task", "research")))

    assert llm.requests == []


def test_worker_error_propagates_unchanged():
    worker = StubWorker(output="unused", error=WorkerError("boom"))
    orchestrator = Orchestrator(build_registry(worker))

    with pytest.raises(WorkerError) as excinfo:
        orchestrator.execute_plan(build_plan(step("step_1", "task", "stub")))

    assert str(excinfo.value) == "boom"


def test_unknown_step_worker_propagates_a_registry_error():
    research = NamedWorker("research")
    orchestrator = Orchestrator(build_registry(research))
    plan = build_plan(step("step_1", "task", "translate"))

    with pytest.raises(RegistryError) as excinfo:
        orchestrator.execute_plan(plan)

    assert type(excinfo.value) is RegistryError
    assert "translate" in str(excinfo.value)
    assert research.calls == []


def test_execution_stops_at_the_first_failing_step():
    log: list[str] = []
    research = RecordingWorker("research", log)
    coding = RecordingWorker("coding", log)
    failing = StubWorker(error=WorkerError("boom"))
    orchestrator = Orchestrator(build_registry(research, failing, coding))
    plan = build_plan(
        step("step_1", "study asyncio", "research"),
        step("step_2", "break here", "stub"),
        step("step_3", "write an example", "coding"),
    )

    with pytest.raises(WorkerError):
        orchestrator.execute_plan(plan)

    assert log == ["research"]
    assert coding.calls == []


def test_plan_steps_are_executed_one_at_a_time():
    events: list[str] = []

    class LifecycleWorker(BaseWorker):
        def __init__(self, name: str) -> None:
            self.name = name

        def execute(self, task: str) -> str:
            events.append(f"{self.name}:start")
            events.append(f"{self.name}:end")
            return f"{self.name} output"

    orchestrator = Orchestrator(
        build_registry(LifecycleWorker("research"), LifecycleWorker("coding"))
    )
    plan = build_plan(
        step("step_1", "study asyncio", "research"),
        step("step_2", "write an example", "coding"),
    )

    orchestrator.execute_plan(plan)

    assert events == [
        "research:start",
        "research:end",
        "coding:start",
        "coding:end",
    ]


def test_single_worker_shorthand_still_supports_execute_plan():
    orchestrator = Orchestrator(StubWorker(output="stub answer"))

    states = orchestrator.execute_plan(build_plan(step("step_1", "task", "stub")))

    assert states[0].final_result == "stub answer"


def test_run_behaviour_is_unchanged_after_phase_4a():
    research = NamedWorker("research", output="research answer")
    orchestrator = Orchestrator(build_registry(research))

    state = orchestrator.run("a question")

    assert state.worker_name == "research"
    assert state.final_result == "research answer"


def test_run_still_wraps_worker_errors():
    worker = StubWorker(error=WorkerError("boom"))
    orchestrator = Orchestrator(build_registry(worker))

    with pytest.raises(OrchestratorError) as excinfo:
        orchestrator.run("task", worker_name="stub")

    assert isinstance(excinfo.value.__cause__, WorkerError)


def test_plan_behaviour_is_unchanged_after_phase_4a():
    expected = build_plan(step("step_1", "task", "research"))
    planner = SpyPlanner(plan=expected)
    orchestrator = Orchestrator(build_registry(NamedWorker("research")), planner=planner)

    plan = orchestrator.plan("do it")

    assert plan is expected
    assert planner.calls == ["do it"]