"""Phase 4E tests: the Evaluator contract and the LLM evaluator."""

from __future__ import annotations

import ast
import inspect
import json
import pathlib

import pytest

from fakes import FakeLLMClient, fake_response
from orchestrator_worker.context import ExecutionContext
from orchestrator_worker.evaluation import EvaluationError, EvaluationResult
from orchestrator_worker.evaluators import Evaluator, EvaluatorError, LLMEvaluator
from orchestrator_worker.execution import ExecutionResult
from orchestrator_worker.llm import LLMError, LLMRequest
from orchestrator_worker.plan import Plan, PlanStep
from orchestrator_worker.state import AgentState

EVALUATORS_PACKAGE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src"
    / "orchestrator_worker"
    / "evaluators"
)


def make_result(
    step_id: str, task: str, worker_name: str, output: str
) -> ExecutionResult:
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


def build_context(
    goal: str,
    specs: list[tuple[str, str, str, str]],
    *,
    result_order: list[str] | None = None,
) -> ExecutionContext:
    steps = [PlanStep(id=sid, task=task, worker_name=wn) for sid, task, wn, _ in specs]
    results = [make_result(*spec) for spec in specs]
    if result_order is not None:
        by_id = {result.step_id: result for result in results}
        results = [by_id[step_id] for step_id in result_order]
    return ExecutionContext(plan=Plan(goal=goal, steps=steps), results=tuple(results))


def single_step_context() -> ExecutionContext:
    return build_context(
        "study asyncio",
        [("step_1", "study the docs", "research", "docs answer")],
    )


def verdict(passed: bool = True, reason: str = "the results are enough") -> str:
    return json.dumps({"passed": passed, "reason": reason})


def imports_of(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))
    return imported


def test_evaluator_is_an_abstract_contract():
    with pytest.raises(TypeError):
        Evaluator()  # type: ignore[abstract]

    assert "evaluate" in Evaluator.__abstractmethods__
    assert list(inspect.signature(Evaluator.evaluate).parameters) == [
        "self",
        "context",
    ]


def test_evaluator_error_is_an_evaluation_error():
    assert issubclass(EvaluatorError, EvaluationError)
    assert issubclass(EvaluatorError, RuntimeError)


def test_llm_evaluator_parses_a_pass():
    llm = FakeLLMClient(fake_response(verdict(True, "everything is covered")))

    result = LLMEvaluator(llm).evaluate(single_step_context())

    assert isinstance(result, EvaluationResult)
    assert result.passed is True
    assert result.reason == "everything is covered"


def test_llm_evaluator_parses_a_failure():
    llm = FakeLLMClient(fake_response(verdict(False, "step_1 returned nothing")))

    result = LLMEvaluator(llm).evaluate(single_step_context())

    assert result.passed is False
    assert result.reason == "step_1 returned nothing"


def test_reason_is_stripped():
    llm = FakeLLMClient(fake_response(verdict(True, "  spaced reason  ")))

    assert LLMEvaluator(llm).evaluate(single_step_context()).reason == "spaced reason"


def test_llm_evaluator_calls_the_llm_client_abstraction():
    llm = FakeLLMClient(fake_response(verdict()))

    LLMEvaluator(llm).evaluate(single_step_context())

    assert len(llm.requests) == 1
    assert isinstance(llm.requests[0], LLMRequest)


def test_llm_evaluator_uses_a_system_and_a_user_message():
    llm = FakeLLMClient(fake_response(verdict()))

    LLMEvaluator(llm).evaluate(single_step_context())

    assert [message.role for message in llm.requests[0].messages] == ["system", "user"]


def test_verdict_does_not_use_json_mode():
    llm = FakeLLMClient(fake_response(verdict()))

    LLMEvaluator(llm).evaluate(single_step_context())

    assert llm.requests[0].json_mode is False


def test_prompt_contains_the_goal_steps_and_outputs():
    llm = FakeLLMClient(fake_response(verdict()))
    context = build_context(
        "study asyncio",
        [
            ("step_1", "study the docs", "research", "docs answer"),
            ("step_2", "write an example", "coding", "code answer"),
        ],
    )

    LLMEvaluator(llm).evaluate(context)

    prompt = llm.requests[0].messages[-1].content
    for expected in [
        "study asyncio",
        "step_1",
        "study the docs",
        "research",
        "docs answer",
        "step_2",
        "write an example",
        "coding",
        "code answer",
        "passed",
    ]:
        assert expected in prompt


def test_prompt_follows_plan_order_even_when_results_are_shuffled():
    llm = FakeLLMClient(fake_response(verdict()))
    context = build_context(
        "a goal",
        [
            ("step_1", "first task", "research", "first output"),
            ("step_2", "second task", "coding", "second output"),
            ("step_3", "third task", "research", "third output"),
        ],
        result_order=["step_3", "step_1", "step_2"],
    )

    LLMEvaluator(llm).evaluate(context)

    prompt = llm.requests[0].messages[-1].content
    assert (
        prompt.index("first task")
        < prompt.index("second task")
        < prompt.index("third task")
    )
    assert (
        prompt.index("first output")
        < prompt.index("second output")
        < prompt.index("third output")
    )
    assert [result.step_id for result in context.results] == [
        "step_3",
        "step_1",
        "step_2",
    ]


def test_evaluator_does_not_modify_the_context():
    llm = FakeLLMClient(fake_response(verdict()))
    context = single_step_context()
    plan_before = context.plan
    steps_before = list(context.plan.steps)
    results_before = context.results

    LLMEvaluator(llm).evaluate(context)

    assert context.plan is plan_before
    assert context.plan.goal == "study asyncio"
    assert list(context.plan.steps) == steps_before
    assert context.results is results_before


def test_malformed_json_is_rejected():
    llm = FakeLLMClient(fake_response("not json at all"))

    with pytest.raises(EvaluationError, match="invalid JSON"):
        LLMEvaluator(llm).evaluate(single_step_context())


@pytest.mark.parametrize("content", ["[]", "[true, false]", '"passed"', "42"])
def test_non_object_json_is_rejected(content):
    llm = FakeLLMClient(fake_response(content))

    with pytest.raises(EvaluationError, match="must be a JSON object"):
        LLMEvaluator(llm).evaluate(single_step_context())


@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
def test_empty_or_whitespace_response_is_rejected(content):
    llm = FakeLLMClient(fake_response(content))

    with pytest.raises(EvaluationError, match="empty response"):
        LLMEvaluator(llm).evaluate(single_step_context())


def test_missing_passed_is_rejected():
    llm = FakeLLMClient(fake_response(json.dumps({"reason": "looks fine"})))

    with pytest.raises(EvaluationError, match="missing required field 'passed'"):
        LLMEvaluator(llm).evaluate(single_step_context())


def test_missing_reason_is_rejected():
    llm = FakeLLMClient(fake_response(json.dumps({"passed": True})))

    with pytest.raises(EvaluationError, match="missing required field 'reason'"):
        LLMEvaluator(llm).evaluate(single_step_context())


@pytest.mark.parametrize("value", [1, 0, "true", None, [], {}])
def test_passed_must_be_a_boolean(value):
    llm = FakeLLMClient(fake_response(json.dumps({"passed": value, "reason": "ok"})))

    with pytest.raises(EvaluationError, match="'passed' must be a boolean"):
        LLMEvaluator(llm).evaluate(single_step_context())


@pytest.mark.parametrize("value", [5, None, True, ["reason"], {}])
def test_reason_must_be_a_string(value):
    llm = FakeLLMClient(fake_response(json.dumps({"passed": True, "reason": value})))

    with pytest.raises(EvaluationError, match="'reason' must be a string"):
        LLMEvaluator(llm).evaluate(single_step_context())


@pytest.mark.parametrize("value", ["", "   ", "\n"])
def test_reason_must_not_be_empty(value):
    llm = FakeLLMClient(fake_response(json.dumps({"passed": False, "reason": value})))

    with pytest.raises(EvaluationError, match="'reason' must not be empty"):
        LLMEvaluator(llm).evaluate(single_step_context())


def test_llm_errors_are_wrapped_and_keep_the_cause():
    error = LLMError("boom")
    llm = FakeLLMClient(error=error)

    with pytest.raises(EvaluationError) as excinfo:
        LLMEvaluator(llm).evaluate(single_step_context())

    assert excinfo.value.__cause__ is error
    assert "boom" in str(excinfo.value)


@pytest.mark.parametrize("value", [None, "a context", [], 42])
def test_evaluate_rejects_values_that_are_not_an_execution_context(value):
    llm = FakeLLMClient(fake_response(verdict()))

    with pytest.raises(EvaluationError, match="expects an ExecutionContext"):
        LLMEvaluator(llm).evaluate(value)


def test_llm_evaluator_module_depends_only_on_project_abstractions():
    assert imports_of(EVALUATORS_PACKAGE / "llm.py") == [
        "__future__",
        "json",
        "typing",
        "..context",
        "..evaluation",
        "..execution",
        "..llm",
        "..plan",
        ".base",
    ]


def test_evaluator_base_module_depends_only_on_context_and_evaluation():
    assert imports_of(EVALUATORS_PACKAGE / "base.py") == [
        "__future__",
        "abc",
        "..context",
        "..evaluation",
    ]


def test_evaluator_package_has_no_provider_worker_router_or_aggregator_dependency():
    for name in ("base.py", "llm.py"):
        imported = imports_of(EVALUATORS_PACKAGE / name)
        assert not any("deepseek" in item or "openai" in item for item in imported)
        assert not any(
            "workers" in item
            or "router" in item
            or "planner" in item
            or "aggregators" in item
            for item in imported
        )