"""Phase 5E-3 tests: evaluator-driven targeted retry.

The retry controller re-executes only the steps the evaluator marked as failed,
in plan order, through the same ``Orchestrator.execute_step`` primitive the
initial plan loop uses. The plan is never modified and never re-planned, and a
failure that cannot be tied to a step fails explicitly instead of re-running the
whole plan.
"""

from __future__ import annotations

import pytest

from fakes import (
    RecordingAggregator,
    RecordingEvaluator,
    RecordingWorker,
    SpyPlanner,
)
from orchestrator_worker.evaluation import EvaluationResult, StepEvaluation
from orchestrator_worker.orchestrator import Orchestrator
from orchestrator_worker.plan import Plan, PlanStep
from orchestrator_worker.retry import RetryError
from orchestrator_worker.workers import BaseWorker, WorkerRegistry


def step(step_id: str, task: str, worker_name: str) -> PlanStep:
    return PlanStep(id=step_id, task=task, worker_name=worker_name)


def build_plan(*steps: PlanStep) -> Plan:
    return Plan(goal="a goal", steps=list(steps))


def build_registry(*workers) -> WorkerRegistry:
    registry = WorkerRegistry()
    for worker in workers:
        registry.register(worker)
    return registry


def passed(reason: str = "ok") -> EvaluationResult:
    return EvaluationResult(passed=True, reason=reason)


def failed(reason: str, *step_ids: str) -> EvaluationResult:
    """A failing verdict whose failed steps are the given ids, in that order."""
    return EvaluationResult(
        passed=False,
        reason=reason,
        step_evaluations=tuple(
            StepEvaluation(step_id=step_id, passed=False, feedback=reason)
            for step_id in step_ids
        ),
    )


class ScriptedWorker(BaseWorker):
    """Worker that returns a new scripted output on each call."""

    def __init__(
        self, name: str, outputs: list[str], log: list[str] | None = None
    ) -> None:
        self.name = name
        self.outputs = list(outputs)
        self.log = log
        self.calls: list[str] = []

    def execute(self, task: str) -> str:
        self.calls.append(task)
        if self.log is not None:
            self.log.append(self.name)
        index = min(len(self.calls) - 1, len(self.outputs) - 1)
        return self.outputs[index]


class StepSpyOrchestrator(Orchestrator):
    """Records the id of every step that goes through ``execute_step``."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.executed_step_ids: list[str] = []

    def execute_step(self, step: PlanStep):
        self.executed_step_ids.append(step.id)
        return super().execute_step(step)


def test_all_steps_pass_means_no_retry():
    log: list[str] = []
    workers = [RecordingWorker(name, log) for name in ("w1", "w2", "w3")]
    evaluator = RecordingEvaluator([passed()])
    orchestrator = StepSpyOrchestrator(build_registry(*workers), evaluator=evaluator)
    plan = build_plan(
        step("step-1", "t1", "w1"),
        step("step-2", "t2", "w2"),
        step("step-3", "t3", "w3"),
    )

    context = orchestrator.execute_plan_with_retry(plan, max_retries=2)

    assert orchestrator.executed_step_ids == ["step-1", "step-2", "step-3"]
    assert log == ["w1", "w2", "w3"]
    assert len(evaluator.calls) == 1
    assert [result.step_id for result in context.results] == [
        "step-1",
        "step-2",
        "step-3",
    ]


def test_single_failed_step_is_the_only_step_re_executed():
    log: list[str] = []
    research = RecordingWorker("research", log)
    coding = RecordingWorker("coding", log)
    evaluator = RecordingEvaluator([failed("step-2 is wrong", "step-2"), passed()])
    orchestrator = StepSpyOrchestrator(
        build_registry(research, coding), evaluator=evaluator
    )
    plan = build_plan(
        step("step-1", "first", "research"),
        step("step-2", "second", "coding"),
        step("step-3", "third", "research"),
    )

    orchestrator.execute_plan_with_retry(plan, max_retries=1)

    assert research.calls == ["first", "third"]
    assert coding.calls == ["second", "second"]
    assert orchestrator.executed_step_ids == [
        "step-1",
        "step-2",
        "step-3",
        "step-2",
    ]
    assert len(evaluator.calls) == 2


def test_multiple_failed_steps_are_all_re_executed():
    log: list[str] = []
    workers = [RecordingWorker(f"w{index}", log) for index in range(1, 5)]
    evaluator = RecordingEvaluator([failed("bad", "step-2", "step-3"), passed()])
    orchestrator = StepSpyOrchestrator(build_registry(*workers), evaluator=evaluator)
    plan = build_plan(
        step("step-1", "t1", "w1"),
        step("step-2", "t2", "w2"),
        step("step-3", "t3", "w3"),
        step("step-4", "t4", "w4"),
    )

    orchestrator.execute_plan_with_retry(plan, max_retries=1)

    assert log == ["w1", "w2", "w3", "w4", "w2", "w3"]
    assert orchestrator.executed_step_ids == [
        "step-1",
        "step-2",
        "step-3",
        "step-4",
        "step-2",
        "step-3",
    ]


def test_retry_stops_as_soon_as_the_results_pass():
    worker = ScriptedWorker("research", ["initial", "retry"])
    evaluator = RecordingEvaluator([failed("bad", "step-1"), passed(), passed()])
    orchestrator = Orchestrator(build_registry(worker), evaluator=evaluator)
    plan = build_plan(step("step-1", "t", "research"))

    context = orchestrator.execute_plan_with_retry(plan, max_retries=5)

    assert len(evaluator.calls) == 2
    assert worker.calls == ["t", "t"]
    assert context.results[0].state.worker_output == "retry"


def test_budget_exhaustion_raises_retry_error_and_stops():
    log: list[str] = []
    worker = RecordingWorker("research", log)
    evaluator = RecordingEvaluator([failed("still bad", "step-1")])
    orchestrator = Orchestrator(build_registry(worker), evaluator=evaluator)
    plan = build_plan(step("step-1", "t", "research"))

    with pytest.raises(RetryError) as excinfo:
        orchestrator.execute_plan_with_retry(plan, max_retries=2)

    assert "after 3 attempt(s)" in str(excinfo.value)
    assert "still bad" in str(excinfo.value)
    assert len(worker.calls) == 3
    assert len(evaluator.calls) == 3


def test_max_retries_zero_does_not_retry_a_targetable_failure():
    log: list[str] = []
    worker = RecordingWorker("research", log)
    evaluator = RecordingEvaluator([failed("bad", "step-1")])
    orchestrator = Orchestrator(build_registry(worker), evaluator=evaluator)
    plan = build_plan(step("step-1", "t", "research"))

    with pytest.raises(RetryError) as excinfo:
        orchestrator.execute_plan_with_retry(plan, max_retries=0)

    assert "after 1 attempt(s)" in str(excinfo.value)
    assert len(worker.calls) == 1
    assert len(evaluator.calls) == 1


def test_planner_is_called_once_across_retries():
    log: list[str] = []
    planner = SpyPlanner(
        plan=build_plan(step("step-1", "t", "research")), log=log
    )
    worker = RecordingWorker("research", log)
    evaluator = RecordingEvaluator([failed("bad", "step-1"), passed()], log=log)
    aggregator = RecordingAggregator(log=log)
    orchestrator = Orchestrator(
        build_registry(worker),
        planner=planner,
        evaluator=evaluator,
        aggregator=aggregator,
    )

    orchestrator.run_pipeline("do it", max_retries=1)

    assert planner.calls == ["do it"]
    assert log.count("planner") == 1


def test_plan_is_not_modified_by_targeted_retry():
    log: list[str] = []
    research = RecordingWorker("research", log)
    coding = RecordingWorker("coding", log)
    evaluator = RecordingEvaluator([failed("bad", "step-2"), passed()])
    orchestrator = Orchestrator(
        build_registry(research, coding), evaluator=evaluator
    )
    plan = build_plan(
        step("step-1", "first", "research"),
        step("step-2", "second", "coding"),
    )
    goal_before = plan.goal
    steps_before = list(plan.steps)

    context = orchestrator.execute_plan_with_retry(plan, max_retries=1)

    assert context.plan is plan
    assert plan.goal == goal_before
    assert list(plan.steps) == steps_before
    assert [step.worker_name for step in plan.steps] == ["research", "coding"]


def test_retry_keeps_the_original_worker_of_the_failed_step():
    log: list[str] = []
    research = RecordingWorker("research", log)
    coding = RecordingWorker("coding", log)
    evaluator = RecordingEvaluator([failed("bad", "step-2"), passed()])
    orchestrator = Orchestrator(
        build_registry(research, coding), evaluator=evaluator
    )
    plan = build_plan(
        step("step-1", "first", "research"),
        step("step-2", "second", "coding"),
    )

    orchestrator.execute_plan_with_retry(plan, max_retries=1)

    assert coding.calls == ["second", "second"]
    assert research.calls == ["first"]
    assert plan.steps[1].worker_name == "coding"


def test_retry_replaces_the_current_result_instead_of_appending():
    worker = ScriptedWorker("research", ["initial", "retry"])
    evaluator = RecordingEvaluator([failed("bad", "step-1"), passed()])
    orchestrator = Orchestrator(build_registry(worker), evaluator=evaluator)
    plan = build_plan(step("step-1", "t", "research"))

    context = orchestrator.execute_plan_with_retry(plan, max_retries=1)

    assert len(context.results) == 1
    assert [result.step_id for result in context.results] == ["step-1"]
    assert context.results[0].state.worker_output == "retry"
    assert evaluator.calls[1] is context


def test_retry_preserves_the_other_steps_results():
    log: list[str] = []
    first = RecordingWorker("w1", log, output="one")
    second = ScriptedWorker("w2", ["two", "two-retry"])
    third = RecordingWorker("w3", log, output="three")
    evaluator = RecordingEvaluator([failed("bad", "step-2"), passed()])
    orchestrator = Orchestrator(
        build_registry(first, second, third), evaluator=evaluator
    )
    plan = build_plan(
        step("step-1", "t1", "w1"),
        step("step-2", "t2", "w2"),
        step("step-3", "t3", "w3"),
    )

    orchestrator.execute_plan_with_retry(plan, max_retries=1)

    initial, updated = evaluator.calls
    assert updated.results[0] is initial.results[0]
    assert updated.results[2] is initial.results[2]
    assert updated.results[1] is not initial.results[1]
    assert updated.results[1].state.worker_output == "two-retry"


def test_retry_order_follows_the_plan_not_the_verdict_order():
    log: list[str] = []
    workers = [RecordingWorker(f"w{index}", log) for index in range(1, 4)]
    evaluator = RecordingEvaluator([failed("bad", "step-3", "step-1"), passed()])
    orchestrator = StepSpyOrchestrator(build_registry(*workers), evaluator=evaluator)
    plan = build_plan(
        step("step-1", "t1", "w1"),
        step("step-2", "t2", "w2"),
        step("step-3", "t3", "w3"),
    )

    orchestrator.execute_plan_with_retry(plan, max_retries=1)

    assert orchestrator.executed_step_ids == [
        "step-1",
        "step-2",
        "step-3",
        "step-1",
        "step-3",
    ]
    assert log == ["w1", "w2", "w3", "w1", "w3"]


def test_each_retry_round_evaluates_once():
    log: list[str] = []
    workers = [RecordingWorker(f"w{index}", log) for index in range(1, 4)]
    evaluator = RecordingEvaluator([failed("bad", "step-2", "step-3"), passed()])
    orchestrator = Orchestrator(build_registry(*workers), evaluator=evaluator)
    plan = build_plan(
        step("step-1", "t1", "w1"),
        step("step-2", "t2", "w2"),
        step("step-3", "t3", "w3"),
    )

    orchestrator.execute_plan_with_retry(plan, max_retries=1)

    assert log == ["w1", "w2", "w3", "w2", "w3"]
    assert len(evaluator.calls) == 2


def test_non_targetable_failure_fails_without_retrying_or_replanning():
    log: list[str] = []
    first = RecordingWorker("w1", log)
    second = RecordingWorker("w2", log)
    planner = SpyPlanner(log=log)
    evaluator = RecordingEvaluator(
        [EvaluationResult(passed=False, reason="cross-step inconsistency")]
    )
    orchestrator = Orchestrator(
        build_registry(first, second),
        planner=planner,
        evaluator=evaluator,
    )
    plan = build_plan(
        step("step-1", "t1", "w1"),
        step("step-2", "t2", "w2"),
    )

    with pytest.raises(RetryError) as excinfo:
        orchestrator.execute_plan_with_retry(plan, max_retries=3)

    assert "no retryable step" in str(excinfo.value)
    assert "cross-step inconsistency" in str(excinfo.value)
    assert log == ["w1", "w2"]
    assert len(evaluator.calls) == 1
    assert planner.calls == []


def test_retry_re_executes_the_original_task_without_feedback_injection():
    """5E-3 keeps the BaseWorker contract unchanged: the same task is re-run."""
    worker = ScriptedWorker("research", ["first", "second"])
    evaluator = RecordingEvaluator([failed("needs work", "step-1"), passed()])
    orchestrator = Orchestrator(build_registry(worker), evaluator=evaluator)
    plan = build_plan(step("step-1", "original task", "research"))

    orchestrator.execute_plan_with_retry(plan, max_retries=1)

    assert worker.calls == ["original task", "original task"]


def test_pipeline_aggregates_once_after_a_successful_retry():
    log: list[str] = []
    planner = SpyPlanner(
        plan=build_plan(step("step-1", "t", "research")), log=log
    )
    worker = RecordingWorker("research", log)
    evaluator = RecordingEvaluator([failed("bad", "step-1"), passed()], log=log)
    aggregator = RecordingAggregator(answer="final answer", log=log)
    orchestrator = Orchestrator(
        build_registry(worker),
        planner=planner,
        evaluator=evaluator,
        aggregator=aggregator,
    )

    assert orchestrator.run_pipeline("task", max_retries=1) == "final answer"
    assert log == [
        "planner",
        "research",
        "evaluator",
        "research",
        "evaluator",
        "aggregator",
    ]
    assert len(aggregator.calls) == 1

def test_unknown_failed_step_id_fails_before_any_retry_execution():
    log: list[str] = []
    first = RecordingWorker("w1", log)
    second = RecordingWorker("w2", log)
    planner = SpyPlanner(log=log)
    evaluator = RecordingEvaluator([failed("bad", "step-999"), passed()], log=log)
    orchestrator = Orchestrator(
        build_registry(first, second),
        planner=planner,
        evaluator=evaluator,
    )
    plan = build_plan(step("step-1", "t1", "w1"), step("step-2", "t2", "w2"))

    with pytest.raises(RetryError) as excinfo:
        orchestrator.execute_plan_with_retry(plan, max_retries=2)

    assert "step-999" in str(excinfo.value)
    assert "not in the plan" in str(excinfo.value)
    assert log == ["w1", "w2", "evaluator"]
    assert first.calls == ["t1"]
    assert second.calls == ["t2"]
    assert len(evaluator.calls) == 1
    assert planner.calls == []
    initial = evaluator.calls[0]
    assert [result.step_id for result in initial.results] == ["step-1", "step-2"]
    assert initial.plan is plan


def test_mixed_known_and_unknown_failed_step_ids_retries_nothing():
    log: list[str] = []
    workers = [RecordingWorker(f"w{index}", log) for index in range(1, 4)]
    evaluator = RecordingEvaluator([failed("bad", "step-999", "step-2"), passed()])
    orchestrator = Orchestrator(build_registry(*workers), evaluator=evaluator)
    plan = build_plan(
        step("step-1", "t1", "w1"),
        step("step-2", "t2", "w2"),
        step("step-3", "t3", "w3"),
    )

    with pytest.raises(RetryError) as excinfo:
        orchestrator.execute_plan_with_retry(plan, max_retries=2)

    assert "step-999" in str(excinfo.value)
    assert "not in the plan" in str(excinfo.value)
    assert log == ["w1", "w2", "w3"]
    assert workers[1].calls == ["t2"]
    assert len(evaluator.calls) == 1
    initial = evaluator.calls[0]
    assert [result.step_id for result in initial.results] == [
        "step-1",
        "step-2",
        "step-3",
    ]
    assert initial.results[1].state.worker_output == "w2 output"


def test_unknown_failed_step_id_is_not_masked_by_a_later_pass():
    log: list[str] = []
    worker = RecordingWorker("research", log)
    evaluator = RecordingEvaluator([failed("bad", "step-999"), passed()])
    orchestrator = Orchestrator(build_registry(worker), evaluator=evaluator)
    plan = build_plan(step("step-1", "t", "research"))

    with pytest.raises(RetryError):
        orchestrator.execute_plan_with_retry(plan, max_retries=1)

    assert len(evaluator.calls) == 1
    assert worker.calls == ["t"]


def test_unknown_failed_step_id_wins_over_an_exhausted_budget():
    log: list[str] = []
    worker = RecordingWorker("research", log)
    evaluator = RecordingEvaluator([failed("bad", "step-999")])
    orchestrator = Orchestrator(build_registry(worker), evaluator=evaluator)
    plan = build_plan(step("step-1", "t", "research"))

    with pytest.raises(RetryError) as excinfo:
        orchestrator.execute_plan_with_retry(plan, max_retries=0)

    assert "not in the plan" in str(excinfo.value)
    assert worker.calls == ["t"]
    assert len(evaluator.calls) == 1