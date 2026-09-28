"""Phase 4D tests: the Aggregator contract and the LLM aggregator."""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

from fakes import FakeLLMClient, fake_response
from orchestrator_worker.aggregators import Aggregator, AggregatorError, LLMAggregator
from orchestrator_worker.context import ExecutionContext
from orchestrator_worker.execution import ExecutionResult
from orchestrator_worker.llm import LLMError, LLMRequest
from orchestrator_worker.plan import Plan, PlanStep
from orchestrator_worker.state import AgentState

AGGREGATOR_PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "src" / "orchestrator_worker" / "aggregators"


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


def imports_of(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))
    return imported


def test_aggregator_is_an_abstract_contract():
    with pytest.raises(TypeError):
        Aggregator()  # type: ignore[abstract]

    assert "aggregate" in Aggregator.__abstractmethods__
    assert list(inspect.signature(Aggregator.aggregate).parameters) == [
        "self",
        "context",
    ]


def test_aggregator_error_is_a_runtime_error():
    assert issubclass(AggregatorError, RuntimeError)


def test_llm_aggregator_returns_the_model_answer():
    llm = FakeLLMClient(fake_response("the final answer"))

    assert LLMAggregator(llm).aggregate(single_step_context()) == "the final answer"


def test_llm_aggregator_calls_the_llm_client_abstraction():
    llm = FakeLLMClient(fake_response("answer"))

    LLMAggregator(llm).aggregate(single_step_context())

    assert len(llm.requests) == 1
    assert isinstance(llm.requests[0], LLMRequest)


def test_llm_aggregator_uses_a_system_and_a_user_message():
    llm = FakeLLMClient(fake_response("answer"))

    LLMAggregator(llm).aggregate(single_step_context())

    messages = llm.requests[0].messages
    assert [message.role for message in messages] == ["system", "user"]


def test_prompt_contains_the_goal_steps_and_results():
    llm = FakeLLMClient(fake_response("answer"))
    context = build_context(
        "study asyncio",
        [
            ("step_1", "study the docs", "research", "docs answer"),
            ("step_2", "write an example", "coding", "code answer"),
        ],
    )

    LLMAggregator(llm).aggregate(context)

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
    ]:
        assert expected in prompt


def test_prompt_follows_plan_order_even_when_results_are_shuffled():
    llm = FakeLLMClient(fake_response("answer"))
    context = build_context(
        "a goal",
        [
            ("step_1", "first task", "research", "first output"),
            ("step_2", "second task", "coding", "second output"),
            ("step_3", "third task", "research", "third output"),
        ],
        result_order=["step_3", "step_1", "step_2"],
    )

    LLMAggregator(llm).aggregate(context)

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


def test_aggregator_does_not_use_json_mode():
    llm = FakeLLMClient(fake_response("answer"))

    LLMAggregator(llm).aggregate(single_step_context())

    assert llm.requests[0].json_mode is False


def test_answer_is_stripped():
    llm = FakeLLMClient(fake_response("  the final answer  \n"))

    assert LLMAggregator(llm).aggregate(single_step_context()) == "the final answer"


@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
def test_empty_model_answer_is_rejected(content):
    llm = FakeLLMClient(fake_response(content))

    with pytest.raises(AggregatorError, match="empty"):
        LLMAggregator(llm).aggregate(single_step_context())


def test_llm_errors_are_wrapped_and_keep_the_cause():
    error = LLMError("boom")
    llm = FakeLLMClient(error=error)

    with pytest.raises(AggregatorError) as excinfo:
        LLMAggregator(llm).aggregate(single_step_context())

    assert excinfo.value.__cause__ is error
    assert "boom" in str(excinfo.value)


@pytest.mark.parametrize("value", [None, "a context", [], 42])
def test_aggregate_rejects_values_that_are_not_an_execution_context(value):
    llm = FakeLLMClient(fake_response("answer"))

    with pytest.raises(AggregatorError, match="expects an ExecutionContext"):
        LLMAggregator(llm).aggregate(value)


def test_aggregator_does_not_modify_the_plan_or_the_context():
    llm = FakeLLMClient(fake_response("answer"))
    context = build_context(
        "a goal",
        [("step_1", "first task", "research", "first output")],
    )
    plan_before = context.plan
    steps_before = list(context.plan.steps)
    results_before = context.results

    LLMAggregator(llm).aggregate(context)

    assert context.plan is plan_before
    assert context.plan.goal == "a goal"
    assert list(context.plan.steps) == steps_before
    assert context.results is results_before


def test_llm_aggregator_module_depends_only_on_project_abstractions():
    assert imports_of(AGGREGATOR_PACKAGE / "llm.py") == [
        "__future__",
        "..context",
        "..execution",
        "..llm",
        "..plan",
        ".base",
    ]


def test_aggregator_package_has_no_provider_planner_router_or_worker_dependency():
    for name in ("base.py", "llm.py"):
        imported = imports_of(AGGREGATOR_PACKAGE / name)
        assert not any("deepseek" in item or "openai" in item for item in imported)
        assert not any(
            "planner" in item or "router" in item or "workers" in item
            for item in imported
        )


def test_aggregator_base_module_imports_only_the_context():
    assert imports_of(AGGREGATOR_PACKAGE / "base.py") == [
        "__future__",
        "abc",
        "..context",
    ]