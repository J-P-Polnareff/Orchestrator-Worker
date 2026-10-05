"""Phase 5E-2 tests: step-aware evaluation and failed step identification."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from fakes import FakeLLMClient, fake_response
from orchestrator_worker.context import ExecutionContext
from orchestrator_worker.evaluation import (
    EvaluationError,
    EvaluationResult,
    StepEvaluation,
)
from orchestrator_worker.evaluators import EvaluatorError, LLMEvaluator
from orchestrator_worker.execution import ExecutionResult
from orchestrator_worker.plan import Plan, PlanStep
from orchestrator_worker.state import AgentState


def build_context(
    goal: str,
    specs: list[tuple[str, str, str, str]],
    *,
    result_order: list[str] | None = None,
) -> ExecutionContext:
    steps = [PlanStep(id=sid, task=task, worker_name=wn) for sid, task, wn, _ in specs]
    results = [_make_result(spec) for spec in specs]
    if result_order is not None:
        by_id = {result.step_id: result for result in results}
        results = [by_id[step_id] for step_id in result_order]
    return ExecutionContext(plan=Plan(goal=goal, steps=steps), results=tuple(results))


def _make_result(spec: tuple[str, str, str, str]) -> ExecutionResult:
    step_id, task, worker_name, output = spec
    return ExecutionResult(
        step_id=step_id,
        task=task,
        worker_name=worker_name,
        state=AgentState(
            user_task=task,
            worker_name=worker_name,
            worker_input=task,
            worker_output=output,
            final_result=output,
        ),
    )


def three_step_context() -> ExecutionContext:
    return build_context(
        "a goal",
        [
            ("step-1", "first task", "research", "first output"),
            ("step-2", "second task", "coding", "second output"),
            ("step-3", "third task", "research", "third output"),
        ],
    )


def step_entry(step_id: str, passed: bool, feedback: str = "") -> dict:
    return {"step_id": step_id, "passed": passed, "feedback": feedback}


def verdict(passed: bool, reason: str, step_evaluations: list[dict]) -> str:
    return json.dumps(
        {
            "passed": passed,
            "reason": reason,
            "step_evaluations": step_evaluations,
        }
    )


def passing(step_ids: tuple[str, ...] = ("step-1", "step-2", "step-3")) -> list[dict]:
    return [step_entry(step_id, True) for step_id in step_ids]


def failing(step_ids: tuple[str, ...]) -> list[dict]:
    return [step_entry(step_id, False, "did not satisfy the requirement") for step_id in step_ids]


# --------------------------------------------------- StepEvaluation value object


def test_step_evaluation_stores_its_fields():
    evaluation = StepEvaluation(step_id="step-1", passed=False, feedback="too short")

    assert evaluation.step_id == "step-1"
    assert evaluation.passed is False
    assert evaluation.feedback == "too short"


def test_step_evaluation_is_frozen():
    evaluation = StepEvaluation(step_id="step-1", passed=True, feedback="")

    with pytest.raises(FrozenInstanceError):
        evaluation.passed = False  # type: ignore[misc]


def test_step_evaluation_allows_empty_feedback():
    assert StepEvaluation(step_id="step-1", passed=True, feedback="").feedback == ""


@pytest.mark.parametrize("value", ["", "   ", "\n\t"])
def test_step_evaluation_rejects_an_empty_step_id(value):
    with pytest.raises(EvaluationError, match="StepEvaluation.step_id"):
        StepEvaluation(step_id=value, passed=True, feedback="")


@pytest.mark.parametrize("value", [None, 123, True])
def test_step_evaluation_rejects_a_non_string_step_id(value):
    with pytest.raises(EvaluationError, match="StepEvaluation.step_id"):
        StepEvaluation(step_id=value, passed=True, feedback="")


@pytest.mark.parametrize("value", ["true", "false", 1, 0, None, []])
def test_step_evaluation_rejects_a_non_bool_passed(value):
    with pytest.raises(EvaluationError, match="StepEvaluation.passed"):
        StepEvaluation(step_id="step-1", passed=value, feedback="")


@pytest.mark.parametrize("value", [None, 123, True, []])
def test_step_evaluation_rejects_a_non_string_feedback(value):
    with pytest.raises(EvaluationError, match="StepEvaluation.feedback"):
        StepEvaluation(step_id="step-1", passed=True, feedback=value)


# --------------------------------------------- EvaluationResult backward compat


def test_evaluation_result_defaults_to_no_step_evaluations():
    result = EvaluationResult(passed=True, reason="ok")

    assert result.step_evaluations == ()
    assert result.failed_step_ids == ()


def test_evaluation_result_still_rejects_invalid_overall_fields():
    with pytest.raises(EvaluationError, match="EvaluationResult.passed"):
        EvaluationResult(passed=1, reason="ok")
    with pytest.raises(EvaluationError, match="EvaluationResult.reason"):
        EvaluationResult(passed=True, reason="")


def test_evaluation_result_rejects_a_non_tuple_step_evaluations():
    step = StepEvaluation(step_id="step-1", passed=True, feedback="")

    with pytest.raises(EvaluationError, match="must be a tuple"):
        EvaluationResult(passed=True, reason="ok", step_evaluations=[step])


def test_evaluation_result_rejects_non_step_evaluation_entries():
    with pytest.raises(EvaluationError, match=r"step_evaluations\[0\]"):
        EvaluationResult(passed=True, reason="ok", step_evaluations=("step-1",))


def test_evaluation_result_rejects_duplicate_step_ids():
    step = StepEvaluation(step_id="step-1", passed=True, feedback="")

    with pytest.raises(EvaluationError, match="duplicate step id"):
        EvaluationResult(passed=True, reason="ok", step_evaluations=(step, step))


# ------------------------------------------------------------- failed_step_ids


def test_failed_step_ids_are_derived_in_order():
    result = EvaluationResult(
        passed=False,
        reason="two steps failed",
        step_evaluations=(
            StepEvaluation("step-1", True, ""),
            StepEvaluation("step-2", False, "bad"),
            StepEvaluation("step-3", False, "worse"),
            StepEvaluation("step-4", True, ""),
        ),
    )

    assert result.failed_step_ids == ("step-2", "step-3")


def test_failed_step_ids_is_empty_when_every_step_passes():
    result = EvaluationResult(
        passed=True,
        reason="all good",
        step_evaluations=(
            StepEvaluation("step-1", True, ""),
            StepEvaluation("step-2", True, ""),
        ),
    )

    assert result.failed_step_ids == ()


def test_failed_step_ids_is_empty_without_step_evaluations():
    assert EvaluationResult(passed=False, reason="global").failed_step_ids == ()


# ------------------------------------------------------------ result invariant


def test_overall_pass_with_a_failed_step_is_rejected():
    with pytest.raises(EvaluationError, match="cannot pass overall"):
        EvaluationResult(
            passed=True,
            reason="looks fine",
            step_evaluations=(
                StepEvaluation("step-1", True, ""),
                StepEvaluation("step-2", False, "problem"),
            ),
        )


def test_overall_failure_without_failed_steps_is_allowed():
    result = EvaluationResult(
        passed=False,
        reason="steps agree but the combination is inconsistent",
        step_evaluations=(
            StepEvaluation("step-1", True, ""),
            StepEvaluation("step-2", True, ""),
        ),
    )

    assert result.passed is False
    assert result.failed_step_ids == ()


# ----------------------------------------------------------- LLMEvaluator prompt


def test_prompt_lists_every_step_id_and_task():
    llm = FakeLLMClient(
        fake_response(verdict(True, "fine", passing()))
    )

    LLMEvaluator(llm).evaluate(three_step_context())

    prompt = llm.requests[0].messages[-1].content
    for expected in ("step-1", "first task", "step-2", "second task", "step-3", "third task"):
        assert expected in prompt


def test_prompt_requires_a_step_by_step_json_verdict():
    llm = FakeLLMClient(fake_response(verdict(True, "fine", passing())))

    LLMEvaluator(llm).evaluate(three_step_context())

    prompt = llm.requests[0].messages[-1].content.lower()
    assert "step_evaluations" in prompt
    assert "step_id" in prompt
    assert "feedback" in prompt
    assert "passed" in prompt
    assert "json" in prompt


# --------------------------------------------------------- LLMEvaluator parsing


def test_valid_step_aware_verdict_is_parsed():
    llm = FakeLLMClient(
        fake_response(
            verdict(
                False,
                "step-2 is not acceptable",
                [
                    step_entry("step-1", True),
                    step_entry("step-2", False, "does not satisfy the requirement"),
                    step_entry("step-3", True),
                ],
            )
        )
    )

    result = LLMEvaluator(llm).evaluate(three_step_context())

    assert isinstance(result, EvaluationResult)
    assert result.passed is False
    assert result.reason == "step-2 is not acceptable"
    assert [entry.step_id for entry in result.step_evaluations] == [
        "step-1",
        "step-2",
        "step-3",
    ]
    assert all(
        isinstance(entry, StepEvaluation) for entry in result.step_evaluations
    )
    assert result.step_evaluations[1].passed is False
    assert result.step_evaluations[1].feedback == "does not satisfy the requirement"


def test_single_failed_step_is_identified():
    llm = FakeLLMClient(
        fake_response(
            verdict(
                False,
                "one step failed",
                [
                    step_entry("step-1", True),
                    step_entry("step-2", False, "wrong answer"),
                    step_entry("step-3", True),
                ],
            )
        )
    )

    assert LLMEvaluator(llm).evaluate(three_step_context()).failed_step_ids == (
        "step-2",
    )


def test_multiple_failed_steps_keep_plan_order():
    llm = FakeLLMClient(
        fake_response(
            verdict(
                False,
                "two steps failed",
                [
                    step_entry("step-3", False, "third bad"),
                    step_entry("step-1", False, "first bad"),
                    step_entry("step-2", True),
                ],
            )
        )
    )

    result = LLMEvaluator(llm).evaluate(three_step_context())

    assert [entry.step_id for entry in result.step_evaluations] == [
        "step-1",
        "step-2",
        "step-3",
    ]
    assert result.failed_step_ids == ("step-1", "step-3")


def test_overall_pass_with_every_step_passing():
    llm = FakeLLMClient(fake_response(verdict(True, "all good", passing())))
    result = LLMEvaluator(llm).evaluate(three_step_context())
    assert result.passed is True
    assert result.failed_step_ids == ()


def test_overall_failure_with_no_failed_step_is_kept():
    llm = FakeLLMClient(
        fake_response(
            verdict(
                False,
                "the steps are individually fine but inconsistent together",
                passing(),
            )
        )
    )

    result = LLMEvaluator(llm).evaluate(three_step_context())

    assert result.passed is False
    assert result.failed_step_ids == ()


def test_evaluator_normalizes_step_order_to_the_plan():
    llm = FakeLLMClient(
        fake_response(
            verdict(
                False,
                "step-2 failed",
                [
                    step_entry("step-3", True),
                    step_entry("step-1", True),
                    step_entry("step-2", False, "bad"),
                ],
            )
        )
    )

    result = LLMEvaluator(llm).evaluate(three_step_context())

    assert [entry.step_id for entry in result.step_evaluations] == [
        "step-1",
        "step-2",
        "step-3",
    ]


# -------------------------------------------------- LLMEvaluator step id checks


def test_unknown_step_id_is_rejected():
    llm = FakeLLMClient(
        fake_response(
            verdict(
                False,
                "bad",
                [
                    step_entry("step-1", True),
                    step_entry("step-2", True),
                    step_entry("step-999", True),
                ],
            )
        )
    )

    with pytest.raises(EvaluationError, match="unknown step"):
        LLMEvaluator(llm).evaluate(three_step_context())


def test_missing_step_id_is_rejected():
    llm = FakeLLMClient(
        fake_response(
            verdict(
                False,
                "bad",
                [step_entry("step-1", True), step_entry("step-2", True)],
            )
        )
    )

    with pytest.raises(EvaluationError, match="missing step evaluations"):
        LLMEvaluator(llm).evaluate(three_step_context())


def test_duplicate_step_id_is_rejected():
    llm = FakeLLMClient(
        fake_response(
            verdict(
                False,
                "bad",
                [
                    step_entry("step-1", True),
                    step_entry("step-2", True),
                    step_entry("step-2", False, "again"),
                    step_entry("step-3", True),
                ],
            )
        )
    )

    with pytest.raises(EvaluationError, match="repeats step id"):
        LLMEvaluator(llm).evaluate(three_step_context())


# ---------------------------------------------------- LLMEvaluator strict schema


def test_missing_step_evaluations_is_rejected():
    llm = FakeLLMClient(fake_response(json.dumps({"passed": True, "reason": "ok"})))

    with pytest.raises(EvaluationError, match="missing required field 'step_evaluations'"):
        LLMEvaluator(llm).evaluate(three_step_context())


@pytest.mark.parametrize("value", [None, {}, "steps", 42])
def test_step_evaluations_must_be_an_array(value):
    llm = FakeLLMClient(
        fake_response(json.dumps({"passed": True, "reason": "ok", "step_evaluations": value}))
    )

    with pytest.raises(EvaluationError, match="'step_evaluations' must be a JSON array"):
        LLMEvaluator(llm).evaluate(three_step_context())


@pytest.mark.parametrize("value", ["step-1", 42, None, ["step-1"]])
def test_step_entry_must_be_an_object(value):
    llm = FakeLLMClient(
        fake_response(verdict(True, "ok", [value, *passing(("step-2", "step-3"))]))
    )

    with pytest.raises(EvaluationError, match=r"step_evaluations\[0\]"):
        LLMEvaluator(llm).evaluate(three_step_context())


@pytest.mark.parametrize("value", ["false", "true", 1, 0, None, []])
def test_step_passed_must_be_a_boolean(value):
    llm = FakeLLMClient(
        fake_response(
            verdict(
                True,
                "ok",
                [
                    {"step_id": "step-1", "passed": value, "feedback": ""},
                    step_entry("step-2", True),
                    step_entry("step-3", True),
                ],
            )
        )
    )

    with pytest.raises(EvaluationError, match="'passed' must be a boolean"):
        LLMEvaluator(llm).evaluate(three_step_context())


@pytest.mark.parametrize("value", [None, 123, True])
def test_step_id_must_be_a_string(value):
    llm = FakeLLMClient(
        fake_response(
            verdict(
                True,
                "ok",
                [
                    {"step_id": value, "passed": True, "feedback": ""},
                    step_entry("step-2", True),
                    step_entry("step-3", True),
                ],
            )
        )
    )

    with pytest.raises(EvaluationError, match="'step_id' must be a string"):
        LLMEvaluator(llm).evaluate(three_step_context())


@pytest.mark.parametrize("value", [None, 123, True, []])
def test_step_feedback_must_be_a_string(value):
    llm = FakeLLMClient(
        fake_response(
            verdict(
                True,
                "ok",
                [
                    {"step_id": "step-1", "passed": True, "feedback": value},
                    step_entry("step-2", True),
                    step_entry("step-3", True),
                ],
            )
        )
    )

    with pytest.raises(EvaluationError, match="'feedback' must be a string"):
        LLMEvaluator(llm).evaluate(three_step_context())


@pytest.mark.parametrize("missing", ["step_id", "passed", "feedback"])
def test_step_entry_missing_field_is_rejected(missing):
    entry = step_entry("step-1", True)
    del entry[missing]
    llm = FakeLLMClient(
        fake_response(verdict(True, "ok", [entry, *passing(("step-2", "step-3"))]))
    )

    with pytest.raises(EvaluationError, match=f"missing required field '{missing}'"):
        LLMEvaluator(llm).evaluate(three_step_context())


def test_overall_pass_with_a_failed_step_in_json_is_rejected():
    llm = FakeLLMClient(
        fake_response(
            verdict(
                True,
                "contradictory",
                [
                    step_entry("step-1", True),
                    step_entry("step-2", False, "bad"),
                    step_entry("step-3", True),
                ],
            )
        )
    )

    with pytest.raises(EvaluationError, match="cannot pass overall"):
        LLMEvaluator(llm).evaluate(three_step_context())


def test_evaluator_does_not_modify_the_context():
    llm = FakeLLMClient(fake_response(verdict(True, "ok", passing())))
    context = three_step_context()
    plan_before = context.plan
    results_before = context.results

    LLMEvaluator(llm).evaluate(context)

    assert context.plan is plan_before
    assert context.results is results_before


def test_evaluator_errors_are_evaluation_errors():
    llm = FakeLLMClient(fake_response("not json"))

    with pytest.raises(EvaluatorError):
        LLMEvaluator(llm).evaluate(three_step_context())

    assert issubclass(EvaluatorError, EvaluationError)