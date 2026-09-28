"""Phase 4A/4B tests: sequential plan execution returning ExecutionResults."""

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
from orchestrator_worker.context import ExecutionContext
from orchestrator_worker.execution import ExecutionResult
from orchestrator_worker.orchestrator import Orchestrator, OrchestratorError
from orchestrator_worker.plan import Plan, PlanStep
from orchestrator_worker.planner import LLMPlanner
from orchestrator_worker.state import AgentState
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

    results = orchestrator.execute_plan(
        build_plan(step("step_1", "study asyncio", "research"))
    )

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, ExecutionResult)
    assert result.step_id == "step_1"
    assert result.task == "study asyncio"
    assert result.worker_name == "research"
    assert isinstance(result.state, AgentState)
    assert result.state.worker_output == "research answer"
    assert result.state.final_result == "research answer"
    assert research.calls == ["study asyncio"]


def test_each_result_records_the_step_input_and_output():
    research = NamedWorker("research", output="research answer")
    orchestrator = Orchestrator(build_registry(research))

    result = orchestrator.execute_plan(
        build_plan(step("step_1", "study asyncio", "research"))
    )[0]

    assert result.state.user_task == "study asyncio"
    assert result.state.worker_input == "study asyncio"
    assert result.state.worker_output == "research answer"


def test_execute_plan_runs_steps_in_plan_order():
    log: list[str] = []
    research = RecordingWorker("research", log)
    coding = RecordingWorker("coding", log)
    orchestrator = Orchestrator(build_registry(research, coding))
    plan = build_plan(
        step("step_1", "study asyncio", "research"),
        step("step_2", "write an example", "coding"),
    )

    results = orchestrator.execute_plan(plan)

    assert log == ["research", "coding"]
    assert research.calls == ["study asyncio"]
    assert coding.calls == ["write an example"]
    assert [result.step_id for result in results] == ["step_1", "step_2"]
    assert [result.task for result in results] == ["study asyncio", "write an example"]
    assert [result.worker_name for result in results] == ["research", "coding"]
    assert len(results) == len(plan.steps)


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

    results = orchestrator.execute_plan(build_plan(step("step_1", "task", "research")))

    assert results[0].state.worker_output == "research answer"


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

    results = orchestrator.execute_plan(build_plan(step("step_1", "task", "stub")))

    assert results[0].state.final_result == "stub answer"


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

def test_each_result_maps_to_its_own_step():
    log: list[str] = []
    research = RecordingWorker("research", log)
    coding = RecordingWorker("coding", log)
    orchestrator = Orchestrator(build_registry(research, coding))
    plan = build_plan(
        step("step_1", "study asyncio", "research"),
        step("step_2", "write an example", "coding"),
        step("step_3", "study more", "research"),
    )

    results = orchestrator.execute_plan(plan)

    assert len(results) == 3
    assert [(r.step_id, r.task, r.worker_name) for r in results] == [
        ("step_1", "study asyncio", "research"),
        ("step_2", "write an example", "coding"),
        ("step_3", "study more", "research"),
    ]
    assert log == ["research", "coding", "research"]


def test_each_result_carries_the_state_its_worker_produced():
    log: list[str] = []
    research = RecordingWorker("research", log, output="research answer")
    coding = RecordingWorker("coding", log, output="coding answer")
    orchestrator = Orchestrator(build_registry(research, coding))
    plan = build_plan(
        step("step_1", "study asyncio", "research"),
        step("step_2", "write an example", "coding"),
    )

    results = orchestrator.execute_plan(plan)

    assert [r.state.worker_output for r in results] == [
        "research answer",
        "coding answer",
    ]
    assert [r.state.final_result for r in results] == [
        "research answer",
        "coding answer",
    ]
    assert [r.state.worker_name for r in results] == ["research", "coding"]
    assert all(isinstance(r.state, AgentState) for r in results)


def test_a_failed_step_raises_instead_of_returning_a_result():
    log: list[str] = []
    research = RecordingWorker("research", log)
    failing = StubWorker(error=WorkerError("boom"))
    orchestrator = Orchestrator(build_registry(research, failing))
    plan = build_plan(
        step("step_1", "study asyncio", "research"),
        step("step_2", "break here", "stub"),
    )

    with pytest.raises(WorkerError):
        orchestrator.execute_plan(plan)

    assert log == ["research"]

def test_execute_plan_context_returns_an_execution_context():
    log: list[str] = []
    orchestrator = Orchestrator(build_registry(RecordingWorker("research", log)))
    plan = build_plan(step("step_1", "study asyncio", "research"))

    context = orchestrator.execute_plan_context(plan)

    assert isinstance(context, ExecutionContext)
    assert context.plan is plan
    assert isinstance(context.results, tuple)
    assert len(context.results) == 1
    assert context.results[0].step_id == "step_1"


def test_execute_plan_context_keeps_every_step_in_order():
    log: list[str] = []
    research = RecordingWorker("research", log)
    coding = RecordingWorker("coding", log)
    orchestrator = Orchestrator(build_registry(research, coding))
    plan = build_plan(
        step("step_1", "study asyncio", "research"),
        step("step_2", "write an example", "coding"),
    )

    context = orchestrator.execute_plan_context(plan)

    assert context.plan is plan
    assert [r.step_id for r in context.results] == ["step_1", "step_2"]
    assert [r.task for r in context.results] == ["study asyncio", "write an example"]
    assert [r.worker_name for r in context.results] == ["research", "coding"]
    assert log == ["research", "coding"]


def test_execute_plan_context_reuses_execute_plan():
    orchestrator = Orchestrator(build_registry(NamedWorker("research")))
    calls: list[Plan] = []
    original = orchestrator.execute_plan

    def recording(plan: Plan) -> list[ExecutionResult]:
        calls.append(plan)
        return original(plan)

    orchestrator.execute_plan = recording  # type: ignore[method-assign]

    plan = build_plan(step("step_1", "task", "research"))
    context = orchestrator.execute_plan_context(plan)

    assert calls == [plan]
    assert [r.step_id for r in context.results] == ["step_1"]


def test_execute_plan_context_does_not_run_workers_twice():
    registry = SpyRegistry()
    registry.register(RecordingWorker("research", []))
    registry.register(RecordingWorker("coding", []))
    orchestrator = Orchestrator(registry)
    plan = build_plan(
        step("step_1", "study asyncio", "research"),
        step("step_2", "write an example", "coding"),
    )

    orchestrator.execute_plan_context(plan)

    assert registry.get_calls == ["research", "coding"]


def test_execute_plan_context_does_not_call_the_planner():
    planner = SpyPlanner()
    orchestrator = Orchestrator(build_registry(NamedWorker("research")), planner=planner)
    plan = build_plan(step("step_1", "task", "research"))

    orchestrator.execute_plan_context(plan)

    assert planner.calls == []


def test_execute_plan_context_never_asks_an_llm():
    llm = FakeLLMClient(fake_response("unused"))
    planner = LLMPlanner(llm, available_workers=["research"])
    orchestrator = Orchestrator(build_registry(NamedWorker("research")), planner=planner)
    plan = build_plan(step("step_1", "task", "research"))

    orchestrator.execute_plan_context(plan)

    assert llm.requests == []


def test_execute_plan_context_propagates_worker_errors():
    worker = StubWorker(error=WorkerError("boom"))
    orchestrator = Orchestrator(build_registry(worker))
    plan = build_plan(step("step_1", "task", "stub"))

    with pytest.raises(WorkerError) as excinfo:
        orchestrator.execute_plan_context(plan)

    assert str(excinfo.value) == "boom"


def test_execute_plan_context_propagates_registry_errors():
    research = NamedWorker("research")
    orchestrator = Orchestrator(build_registry(research))
    plan = build_plan(step("step_1", "task", "translate"))

    with pytest.raises(RegistryError) as excinfo:
        orchestrator.execute_plan_context(plan)

    assert type(excinfo.value) is RegistryError
    assert research.calls == []


def test_execute_plan_still_returns_a_list():
    orchestrator = Orchestrator(build_registry(NamedWorker("research")))

    results = orchestrator.execute_plan(build_plan(step("step_1", "task", "research")))

    assert isinstance(results, list)
    assert len(results) == 1