"""Phase 5E-1 tests: single-step execution and context result replacement."""

from __future__ import annotations

import pytest

from fakes import (
    FakeLLMClient,
    NamedWorker,
    SequenceLLMClient,
    SpyRegistry,
    StubWorker,
    fake_response,
)
from orchestrator_worker.builtin_tools import calculator_tool
from orchestrator_worker.context import ExecutionContext, ExecutionContextError
from orchestrator_worker.execution import ExecutionResult
from orchestrator_worker.llm import LLMResponse
from orchestrator_worker.orchestrator import Orchestrator
from orchestrator_worker.plan import Plan, PlanStep
from orchestrator_worker.planner import LLMPlanner
from orchestrator_worker.state import AgentState
from orchestrator_worker.tool_loop import ToolLoop
from orchestrator_worker.tools import ToolCall, ToolExecutor, ToolRegistry
from orchestrator_worker.workers import RegistryError, WorkerError, WorkerRegistry
from orchestrator_worker.workers.research_tool import ResearchToolWorker


def step(step_id: str, task: str = "a task", worker_name: str = "research") -> PlanStep:
    return PlanStep(id=step_id, task=task, worker_name=worker_name)


def build_plan(*steps: PlanStep) -> Plan:
    return Plan(goal="a goal", steps=list(steps))


def build_registry(*workers) -> WorkerRegistry:
    registry = WorkerRegistry()
    for worker in workers:
        registry.register(worker)
    return registry


def make_plan(*step_ids: str) -> Plan:
    return Plan(
        goal="a goal",
        steps=[
            PlanStep(id=step_id, task=f"task for {step_id}", worker_name="research")
            for step_id in step_ids
        ],
    )


def make_result(step_id: str, output: str = "done") -> ExecutionResult:
    return ExecutionResult(
        step_id=step_id,
        task=f"task for {step_id}",
        worker_name="research",
        state=AgentState(user_task=f"task for {step_id}", final_result=output),
    )


def make_context(plan: Plan, *step_ids: str) -> ExecutionContext:
    return ExecutionContext(plan=plan, results=tuple(make_result(i) for i in step_ids))


# ----------------------------------------------------- execute_step primitive


def test_execute_step_runs_only_the_requested_step():
    research = NamedWorker("research", output="research answer")
    coding = NamedWorker("coding", output="coding answer")
    orchestrator = Orchestrator(build_registry(research, coding))

    result = orchestrator.execute_step(step("step_1", "study asyncio", "research"))

    assert isinstance(result, ExecutionResult)
    assert result.step_id == "step_1"
    assert result.task == "study asyncio"
    assert result.worker_name == "research"
    assert result.state.worker_output == "research answer"
    assert result.state.final_result == "research answer"
    assert research.calls == ["study asyncio"]
    assert coding.calls == []


def test_execute_step_records_the_worker_input_and_output():
    research = NamedWorker("research", output="research answer")
    orchestrator = Orchestrator(build_registry(research))

    result = orchestrator.execute_step(step("step_1", "study asyncio", "research"))

    assert result.state.user_task == "study asyncio"
    assert result.state.worker_name == "research"
    assert result.state.worker_input == "study asyncio"
    assert isinstance(result.state, AgentState)


def test_execute_step_routes_through_the_registry_once():
    registry = SpyRegistry()
    registry.register(NamedWorker("research"))
    orchestrator = Orchestrator(registry)

    orchestrator.execute_step(step("step_1"))

    assert registry.get_calls == ["research"]


def test_execute_step_propagates_worker_errors():
    worker = StubWorker(error=WorkerError("boom"))
    orchestrator = Orchestrator(build_registry(worker))

    with pytest.raises(WorkerError) as excinfo:
        orchestrator.execute_step(step("step_1", "task", "stub"))

    assert str(excinfo.value) == "boom"


def test_execute_step_propagates_registry_errors_without_fallback():
    research = NamedWorker("research")
    orchestrator = Orchestrator(build_registry(research))

    with pytest.raises(RegistryError):
        orchestrator.execute_step(step("step_1", "task", "translate"))

    assert research.calls == []


def test_execute_step_does_not_add_llm_calls_for_routing():
    llm = FakeLLMClient(fake_response("unused"))
    planner = LLMPlanner(llm, available_workers=["research"])
    orchestrator = Orchestrator(build_registry(NamedWorker("research")), planner=planner)

    orchestrator.execute_step(step("step_1"))

    assert llm.requests == []


def test_execute_step_runs_a_tool_enabled_worker_through_its_tool_loop():
    calculator = calculator_tool()
    tools = ToolRegistry()
    tools.register(calculator)
    llm = SequenceLLMClient(
        [
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
        ]
    )
    loop = ToolLoop(llm, ToolExecutor(tools))
    worker = ResearchToolWorker(loop, tools=(calculator,))

    orchestrator = Orchestrator(build_registry(worker))
    result = orchestrator.execute_step(step("step_1", "compute 17 x 23", "research"))

    assert result.state.worker_output == "17 x 23 = 391."
    assert len(llm.requests) == 2
    follow_up = llm.requests[1]
    assert [message.role for message in follow_up.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert follow_up.messages[-1].content == "391"


# ------------------------------------- execute_plan reuses the same primitive


def test_execute_plan_dispatches_every_step_through_execute_step(monkeypatch):
    calls: list[str] = []
    original = Orchestrator.execute_step

    def recording(self, plan_step: PlanStep) -> ExecutionResult:
        calls.append(plan_step.id)
        return original(self, plan_step)

    monkeypatch.setattr(Orchestrator, "execute_step", recording)

    orchestrator = Orchestrator(build_registry(NamedWorker("research")))
    plan = build_plan(step("step_1"), step("step_2"))

    results = orchestrator.execute_plan(plan)

    assert calls == ["step_1", "step_2"]
    assert [result.step_id for result in results] == ["step_1", "step_2"]


def test_execute_plan_runs_each_worker_once():
    research = NamedWorker("research", output="research answer")
    orchestrator = Orchestrator(build_registry(research))

    orchestrator.execute_plan(build_plan(step("step_1"), step("step_2")))

    assert research.calls == ["a task", "a task"]


def test_execute_plan_results_match_execute_step_results():
    research = NamedWorker("research", output="research answer")
    orchestrator = Orchestrator(build_registry(research))
    plan_step = step("step_1", "study asyncio", "research")

    via_plan = orchestrator.execute_plan(build_plan(plan_step))[0]
    via_step = orchestrator.execute_step(plan_step)

    assert via_plan.step_id == via_step.step_id
    assert via_plan.task == via_step.task
    assert via_plan.worker_name == via_step.worker_name
    assert via_plan.state.worker_output == via_step.state.worker_output
    assert via_plan.state.final_result == via_step.state.final_result


# --------------------------------------- replacing a step's current result


def test_replace_step_result_swaps_the_current_result():
    plan = make_plan("step_1", "step_2", "step_3")
    context = make_context(plan, "step_1", "step_2", "step_3")
    replacement = make_result("step_2", output="result-D")

    updated = context.replace_step_result("step_2", replacement)

    assert isinstance(updated, ExecutionContext)
    assert updated.plan is plan
    assert [result.step_id for result in updated.results] == [
        "step_1",
        "step_2",
        "step_3",
    ]
    assert updated.results[1] is replacement
    assert updated.results[1].state.final_result == "result-D"
    assert [result.step_id for result in updated.results].count("step_2") == 1


def test_replace_step_result_leaves_other_steps_untouched():
    plan = make_plan("step_1", "step_2", "step_3")
    context = make_context(plan, "step_1", "step_2", "step_3")

    updated = context.replace_step_result("step_2", make_result("step_2", "D"))

    assert updated.results[0] is context.results[0]
    assert updated.results[2] is context.results[2]
    assert updated.plan is context.plan


def test_replace_step_result_does_not_mutate_the_original_context():
    plan = make_plan("step_1", "step_2")
    context = make_context(plan, "step_1", "step_2")
    before = context.results

    context.replace_step_result("step_2", make_result("step_2", "D"))

    assert context.results is before
    assert context.results[1].state.final_result == "done"


def test_replace_step_result_rejects_an_unknown_step():
    plan = make_plan("step_1", "step_2")
    context = make_context(plan, "step_1", "step_2")

    with pytest.raises(ExecutionContextError, match="unknown step"):
        context.replace_step_result("step_3", make_result("step_3"))


def test_replace_step_result_does_not_append_an_unknown_step():
    plan = make_plan("step_1", "step_2")
    context = make_context(plan, "step_1", "step_2")

    with pytest.raises(ExecutionContextError):
        context.replace_step_result("step_3", make_result("step_3"))

    assert [result.step_id for result in context.results] == ["step_1", "step_2"]


def test_replace_step_result_rejects_a_result_for_a_different_step():
    plan = make_plan("step_1", "step_2")
    context = make_context(plan, "step_1", "step_2")

    with pytest.raises(ExecutionContextError, match="cannot replace"):
        context.replace_step_result("step_2", make_result("step_1"))


def test_replace_step_result_rejects_a_non_result():
    context = make_context(make_plan("step_1"), "step_1")

    with pytest.raises(ExecutionContextError, match="must be an ExecutionResult"):
        context.replace_step_result("step_1", "not-a-result")


@pytest.mark.parametrize("step_id", ["", "   "])
def test_replace_step_result_requires_a_non_empty_step_id(step_id):
    context = make_context(make_plan("step_1"), "step_1")

    with pytest.raises(ExecutionContextError, match="non-empty string"):
        context.replace_step_result(step_id, make_result("step_1"))


def test_replace_step_result_uses_step_id_not_worker_name():
    plan = Plan(
        goal="a goal",
        steps=[
            PlanStep(id="step_1", task="first", worker_name="research"),
            PlanStep(id="step_2", task="second", worker_name="research"),
        ],
    )
    context = ExecutionContext(
        plan=plan,
        results=(make_result("step_1"), make_result("step_2")),
    )
    replacement = make_result("step_2", output="second-run")

    updated = context.replace_step_result("step_2", replacement)

    assert updated.results[0] is context.results[0]
    assert updated.results[0].step_id == "step_1"
    assert updated.results[1] is replacement