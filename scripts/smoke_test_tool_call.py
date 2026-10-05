r"""Manual, cost-controlled real-DeepSeek tool-calling smoke test.

Run it by hand only:

    .\.venv\Scripts\python.exe scripts\smoke_test_tool_call.py

It is deliberately NOT a pytest test: it makes real API calls, so it never runs
as part of the suite. It goes through the Phase 5A composition root
(``build_orchestrator()`` with ``llm=None``) and the Phase 4F ``run_pipeline()``
with ``max_retries=0``, so it drives the real DeepSeek planner, the real
``ToolLoop`` + ``ToolExecutor`` + calculator, the real evaluator and the real
aggregator.

Everything it touches is harness-local and lives only for this process:

* a recording LLM client wrapped around the one shared client, so every real
  request/response pair is observed without changing a production module;
* a planner wrapper that records the plan and stops before any worker runs when
  the plan is not exactly one ``research`` step (API-cost guard).

The API key is never printed - only whether it is present.
"""

from __future__ import annotations

from dataclasses import dataclass

from smoke_test import _stage_of

from orchestrator_worker.application import build_orchestrator
from orchestrator_worker.config import Settings
from orchestrator_worker.llm import LLMClient, LLMRequest, LLMResponse
from orchestrator_worker.plan import Plan
from orchestrator_worker.planner import Planner

TASK = (
    "请作为研究员完成一次简单的数值事实核验：使用你可用的 calculator 工具验证 17 × 23 的结果，"
    "并用中文一句话报告核验结果。整个任务只需要一个 research worker 步骤；"
    "calculator 是 research worker 在执行过程中使用的内部工具，不需要 coding worker，"
    "不需要编写代码，也不需要其他工具。"
    "你必须实际调用 calculator，不能心算或猜测结果。"
)

CALCULATOR_ARGUMENTS = {"a": 17, "b": 23, "operation": "multiply"}
EXPECTED_RESULT = "391"


class HarnessError(RuntimeError):
    """Raised by the harness when a smoke-test precondition is violated."""


@dataclass
class _Call:
    """One real LLM call: the stage that made it, the request and the response."""

    stage: str
    request: LLMRequest
    response: LLMResponse | None = None


class _RecordingClient(LLMClient):
    """Delegates to the real client and records every call it observes."""

    name = "tool-smoke-recording"

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.calls: list[_Call] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        call = _Call(stage=_stage_of(request), request=request)
        self.calls.append(call)
        call.response = self._inner.complete(request)
        return call.response


class _GuardedPlanner(Planner):
    """Harness seam: real planner, but stop before workers on a bad plan."""

    def __init__(self, inner: Planner) -> None:
        self._inner = inner
        self.plan: Plan | None = None
        self.calls = 0

    def create_plan(self, task: str) -> Plan:
        self.calls += 1
        plan = self._inner.create_plan(task)
        self.plan = plan
        if len(plan.steps) != 1:
            raise HarnessError(
                f"Planner produced unexpected number of steps: {len(plan.steps)}"
            )
        if plan.steps[0].worker_name != "research":
            raise HarnessError(
                f"Unexpected worker_name: {plan.steps[0].worker_name!r}"
            )
        return plan


def _install_recorder(orchestrator) -> _RecordingClient:
    """Wrap the single shared LLM client so every real call is observed."""
    proxy = _RecordingClient(orchestrator._planner._llm)

    orchestrator._planner._llm = proxy
    orchestrator._evaluator._llm = proxy
    orchestrator._aggregator._llm = proxy
    for name in orchestrator._registry.names():
        worker = orchestrator._registry.get(name)
        if hasattr(worker, "_llm"):
            worker._llm = proxy
        if hasattr(worker, "_tool_loop"):
            worker._tool_loop._llm = proxy
    return proxy


def _stages(recorder: _RecordingClient) -> list[str]:
    return [call.stage for call in recorder.calls]


def _print_plan(planner: _GuardedPlanner) -> None:
    if planner.plan is None:
        print("Planner step count: (unknown)")
        return
    print(f"Planner step count: {len(planner.plan.steps)}")
    for step in planner.plan.steps:
        print(f"  step {step.id}: worker={step.worker_name!r} task={step.task!r}")


def _print_progress(recorder: _RecordingClient) -> None:
    stages = _stages(recorder)
    print(f"Observed stages: {' -> '.join(stages) if stages else '(none)'}")
    print(
        "Stage counts: "
        f"planner={stages.count('planner')} "
        f"research={stages.count('research')} "
        f"evaluator={stages.count('evaluator')} "
        f"aggregator={stages.count('aggregator')}"
    )


def _print_tool_flow(recorder: _RecordingClient) -> None:
    research = [call for call in recorder.calls if call.stage == "research"]
    if research and research[0].response is not None:
        print("First research tool_calls:")
        for call in research[0].response.tool_calls:
            print(
                f"  name={call.name!r} id={call.id!r} "
                f"arguments={call.arguments!r}"
            )
    if len(research) > 1:
        second = research[1]
        print(
            "Second research request roles: "
            + " -> ".join(message.role for message in second.request.messages)
        )
        for message in second.request.messages:
            if message.role == "tool":
                print(
                    f"  tool message: id={message.tool_call_id!r} "
                    f"content={message.content!r}"
                )
        if second.response is not None:
            print(f"Research final response: {second.response.content!r}")


def _verify(
    recorder: _RecordingClient,
    planner: _GuardedPlanner,
    answer: str,
) -> list[str]:
    """Return every unmet success condition (empty list means all good)."""
    problems: list[str] = []
    stages = _stages(recorder)

    if planner.calls != 1:
        problems.append(f"planner called {planner.calls} times, expected 1")
    if stages.count("planner") != 1:
        problems.append(f"planner LLM calls {stages.count('planner')}, expected 1")
    if stages.count("evaluator") < 1:
        problems.append("evaluator was never called")
    if stages.count("aggregator") < 1:
        problems.append("aggregator was never called")
    if not (answer or "").strip():
        problems.append("run_pipeline returned an empty final answer")

    research = [call for call in recorder.calls if call.stage == "research"]
    if len(research) < 2:
        problems.append(f"expected at least 2 research LLM calls, saw {len(research)}")
        return problems

    first, second = research[0], research[1]
    tool_calls = first.response.tool_calls if first.response is not None else ()

    if not tool_calls:
        problems.append("Real DeepSeek did not produce a tool call.")
        return problems

    calculators = [call for call in tool_calls if call.name == "calculator"]
    if not calculators:
        problems.append(
            "no calculator tool call; got "
            + ", ".join(repr(call.name) for call in tool_calls)
        )
        return problems

    calculator = calculators[0]
    if calculator.arguments != CALCULATOR_ARGUMENTS:
        problems.append(
            f"calculator arguments {calculator.arguments!r} "
            f"!= {CALCULATOR_ARGUMENTS!r}"
        )

    messages = list(second.request.messages)
    assistant_indexes = [
        index
        for index, message in enumerate(messages)
        if message.role == "assistant" and message.tool_calls
    ]
    tool_indexes = [
        index for index, message in enumerate(messages) if message.role == "tool"
    ]

    if not assistant_indexes:
        problems.append("second research request has no assistant tool-call message")
    if not tool_indexes:
        problems.append("second research request has no tool message")
    elif assistant_indexes and tool_indexes[0] < assistant_indexes[0]:
        problems.append("tool message appears before the assistant tool-call message")

    if tool_indexes:
        tool_message = messages[tool_indexes[0]]
        if tool_message.tool_call_id != calculator.id:
            problems.append(
                f"tool_call_id {tool_message.tool_call_id!r} != {calculator.id!r}"
            )
        if tool_message.content != EXPECTED_RESULT:
            problems.append(
                f"tool result {tool_message.content!r} != {EXPECTED_RESULT!r}"
            )

    if not any(tool.name == "calculator" for tool in second.request.tools):
        problems.append("second research request dropped the calculator definition")

    if second.response is not None:
        if second.response.tool_calls:
            problems.append("second research response still requested tools")
        content = (second.response.content or "").strip()
        if not content:
            problems.append("second research response was empty")
        elif EXPECTED_RESULT not in content:
            problems.append(
                f"worker answer does not mention {EXPECTED_RESULT}: {content!r}"
            )

    return problems


def main() -> int:
    print("Smoke test tool call started")

    settings = Settings.from_env()
    if not settings.api_key:
        print("DEEPSEEK_API_KEY is missing.")
        return 1
    print("DEEPSEEK_API_KEY is configured.")
    print(f"Task: {TASK}")

    orchestrator = build_orchestrator()

    recorder = _install_recorder(orchestrator)
    planner = _GuardedPlanner(orchestrator._planner)
    orchestrator._planner = planner

    try:
        answer = orchestrator.run_pipeline(TASK, max_retries=0)
    except HarnessError as exc:
        _print_plan(planner)
        _print_progress(recorder)
        print(f"HarnessError: {exc}")
        return 1
    except Exception as exc:  # report and stop - never catch and retry
        _print_plan(planner)
        _print_progress(recorder)
        print(f"{type(exc).__name__}: {exc}")
        return 1

    _print_plan(planner)
    _print_progress(recorder)
    _print_tool_flow(recorder)
    print(f"Total LLM calls observed: {len(recorder.calls)}")

    problems = _verify(recorder, planner, answer)
    if problems:
        print("Smoke test tool call FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("Final answer:")
    print(answer)
    print("Smoke test tool call succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())