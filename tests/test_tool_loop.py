"""Phase 5C-3 tests: the provider-neutral tool execution loop.

Everything here is offline. The LLM double is scripted, the tools are plain
callables and no provider SDK is imported.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

from orchestrator_worker.llm import (
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
)
from orchestrator_worker.tool_loop import ToolLoop, ToolLoopError
from orchestrator_worker.tools import (
    Tool,
    ToolCall,
    ToolExecutor,
    ToolRegistry,
    ToolRegistryError,
    ToolResult,
)

PARAMETERS = {
    "type": "object",
    "properties": {
        "a": {"type": "number"},
        "b": {"type": "number"},
    },
    "required": ["a", "b"],
}


class ScriptedLLMClient(LLMClient):
    """Returns scripted responses in order; an exception entry is raised."""

    name = "scripted"

    def __init__(self, *script: object) -> None:
        self.script = list(script)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        index = len(self.requests)
        self.requests.append(request)
        if index >= len(self.script):
            raise AssertionError(
                f"unexpected LLM call #{index + 1}; the script has "
                f"{len(self.script)} entries."
            )
        entry = self.script[index]
        if isinstance(entry, BaseException):
            raise entry
        return entry

    @property
    def calls(self) -> int:
        return len(self.requests)


class RecordingToolExecutor(ToolExecutor):
    """ToolExecutor that records every call it is handed."""

    def __init__(self, registry: ToolRegistry) -> None:
        super().__init__(registry)
        self.calls: list[ToolCall] = []

    def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        return super().execute(call)


class ResultCapturingToolLoop(ToolLoop):
    """Test seam: records each ToolResult the loop builds for the model."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.results: list[ToolResult] = []

    def _execute(self, call: ToolCall) -> ToolResult:
        result = super()._execute(call)
        self.results.append(result)
        return result


def calculator_tool(handler=None) -> Tool:
    return Tool(
        name="calculator",
        description="Perform a simple arithmetic operation.",
        parameters=PARAMETERS,
        handler=(
            handler
            if handler is not None
            else (lambda args: str(args["a"] + args["b"]))
        ),
    )


def failing_tool(tool_name: str = "failing", message: str = "boom") -> Tool:
    def handler(args):
        raise ValueError(message)

    return Tool(
        name=tool_name,
        description="Always fails.",
        parameters={},
        handler=handler,
    )


def build_executor(*tools: Tool) -> RecordingToolExecutor:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return RecordingToolExecutor(registry)


def tool_call(call_id: str = "call-1", name: str = "calculator", arguments=None) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=name,
        arguments={"a": 1, "b": 2} if arguments is None else arguments,
    )


def tool_call_response(*calls: ToolCall, content: str = "") -> LLMResponse:
    return LLMResponse(content=content, model="fake-model", tool_calls=tuple(calls))


def final_response(content: str = "final answer") -> LLMResponse:
    return LLMResponse(content=content, model="fake-model")


def make_request(tools=(), messages=None, **kwargs) -> LLMRequest:
    return LLMRequest(
        messages=messages if messages is not None else (Message.user("go"),),
        tools=tuple(tools),
        **kwargs,
    )

# ----------------------------------------------------------------- construction


def test_tool_loop_error_is_a_runtime_error():
    assert issubclass(ToolLoopError, RuntimeError)


def test_tool_loop_is_a_class():
    assert isinstance(ToolLoop, type)


@pytest.mark.parametrize("bad", [object(), None, "llm", 42])
def test_rejects_a_non_llm_client(bad):
    with pytest.raises(ToolLoopError, match="expects an LLMClient"):
        ToolLoop(bad, build_executor(calculator_tool()))


@pytest.mark.parametrize("bad", [object(), None, "executor", 42])
def test_rejects_a_non_tool_executor(bad):
    with pytest.raises(ToolLoopError, match="expects a ToolExecutor"):
        ToolLoop(ScriptedLLMClient(), bad)


@pytest.mark.parametrize("bad", ["2", 1.5, None, [2], object()])
def test_rejects_non_int_max_rounds(bad):
    with pytest.raises(ToolLoopError, match="max_rounds must be an int"):
        ToolLoop(
            ScriptedLLMClient(),
            build_executor(calculator_tool()),
            max_rounds=bad,
        )


@pytest.mark.parametrize("bad", [True, False])
def test_rejects_bool_max_rounds(bad):
    with pytest.raises(ToolLoopError, match="max_rounds must be an int"):
        ToolLoop(
            ScriptedLLMClient(),
            build_executor(calculator_tool()),
            max_rounds=bad,
        )


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_rejects_out_of_range_max_rounds(bad):
    with pytest.raises(ToolLoopError, match="must be at least 1"):
        ToolLoop(
            ScriptedLLMClient(),
            build_executor(calculator_tool()),
            max_rounds=bad,
        )


def test_run_rejects_a_non_request():
    loop = ToolLoop(ScriptedLLMClient(), build_executor(calculator_tool()))

    with pytest.raises(ToolLoopError, match="expects an LLMRequest"):
        loop.run("not a request")


# --------------------------------------------------------------- no tool calls


def test_final_response_costs_one_llm_call():
    llm = ScriptedLLMClient(final_response("done"))
    loop = ToolLoop(llm, build_executor(calculator_tool()))

    response = loop.run(make_request((calculator_tool(),)))

    assert response.content == "done"
    assert llm.calls == 1


def test_final_response_is_returned_unchanged():
    response = final_response("done")
    llm = ScriptedLLMClient(response)
    loop = ToolLoop(llm, build_executor(calculator_tool()))

    assert loop.run(make_request((calculator_tool(),))) is response


def test_no_tool_call_means_no_tool_execution():
    executor = build_executor(calculator_tool())
    loop = ToolLoop(ScriptedLLMClient(final_response()), executor)

    loop.run(make_request((calculator_tool(),)))

    assert executor.calls == []


def test_no_tool_call_means_no_extra_history_or_call():
    llm = ScriptedLLMClient(final_response())
    request = make_request((calculator_tool(),))

    ToolLoop(llm, build_executor(calculator_tool())).run(request)

    assert llm.requests == [request]
    assert llm.requests[0].messages is request.messages


# ------------------------------------------------------------ single tool call


def test_single_tool_call_runs_to_a_final_answer():
    call = tool_call()
    llm = ScriptedLLMClient(tool_call_response(call), final_response("42"))
    executor = build_executor(calculator_tool())

    response = ToolLoop(llm, executor).run(make_request((calculator_tool(),)))

    assert response.content == "42"
    assert llm.calls == 2
    assert executor.calls == [call]


def test_executor_receives_the_original_tool_call_object():
    call = tool_call()
    executor = build_executor(calculator_tool())
    llm = ScriptedLLMClient(tool_call_response(call), final_response())

    ToolLoop(llm, executor).run(make_request((calculator_tool(),)))

    assert executor.calls[0] is call


def test_arguments_reach_the_handler_unchanged():
    seen = []

    def handler(args):
        seen.append(args)
        return "ok"

    tool = calculator_tool(handler=handler)
    call = tool_call(arguments={"a": 7, "b": 35})
    llm = ScriptedLLMClient(tool_call_response(call), final_response())

    ToolLoop(llm, build_executor(tool)).run(make_request((tool,)))

    assert seen == [{"a": 7, "b": 35}]
    assert seen[0] is call.arguments


def test_assistant_tool_call_message_is_added():
    call = tool_call()
    llm = ScriptedLLMClient(
        tool_call_response(call, content="computing"), final_response()
    )
    tool = calculator_tool()

    ToolLoop(llm, build_executor(tool)).run(
        make_request((tool,), messages=(Message.user("add 1 and 2"),))
    )

    assistant = llm.requests[1].messages[1]
    assert assistant.role == "assistant"
    assert assistant.content == "computing"
    assert assistant.tool_calls == (call,)


def test_tool_result_message_is_added():
    call = tool_call()
    llm = ScriptedLLMClient(tool_call_response(call), final_response())
    tool = calculator_tool()

    ToolLoop(llm, build_executor(tool)).run(make_request((tool,)))

    tool_message = llm.requests[1].messages[2]
    assert tool_message.role == "tool"
    assert tool_message.tool_call_id == "call-1"
    assert tool_message.content == "3"


def test_follow_up_request_keeps_the_original_head():
    call = tool_call()
    llm = ScriptedLLMClient(tool_call_response(call), final_response())
    tool = calculator_tool()

    ToolLoop(llm, build_executor(tool)).run(
        make_request((tool,), messages=(Message.user("add 1 and 2"),))
    )

    assert llm.requests[1].messages[0] == Message.user("add 1 and 2")


def test_the_second_response_is_returned():
    expected = final_response("done")
    llm = ScriptedLLMClient(tool_call_response(tool_call()), expected)
    tool = calculator_tool()

    assert ToolLoop(llm, build_executor(tool)).run(make_request((tool,))) is expected

# --------------------------------------------------------- multiple tool calls


def test_every_tool_call_in_one_response_executes():
    calls = (tool_call("call-1"), tool_call("call-2"), tool_call("call-3"))
    llm = ScriptedLLMClient(tool_call_response(*calls), final_response())
    executor = build_executor(calculator_tool())

    ToolLoop(llm, executor).run(make_request((calculator_tool(),)))

    assert executor.calls == list(calls)


def test_tool_calls_execute_in_response_order():
    log: list = []

    def make_handler(name):
        def handler(args):
            log.append(name)
            return name

        return handler

    tools = tuple(
        Tool(name=n, description="d", parameters={}, handler=make_handler(n))
        for n in ("a", "b", "c")
    )
    calls = tuple(
        ToolCall(id=f"call-{n}", name=n, arguments={}) for n in ("a", "b", "c")
    )
    llm = ScriptedLLMClient(tool_call_response(*calls), final_response())

    ToolLoop(llm, build_executor(*tools)).run(make_request(tools))

    assert log == ["a", "b", "c"]


def test_tool_calls_are_sequential_and_never_overlap():
    events: list = []

    def make_handler(name: str):
        def handler(args):
            events.append(f"{name}:start")
            events.append(f"{name}:end")
            return name

        return handler

    tools = tuple(
        Tool(name=n, description="d", parameters={}, handler=make_handler(n))
        for n in ("a", "b", "c")
    )
    calls = tuple(
        ToolCall(id=f"call-{n}", name=n, arguments={}) for n in ("a", "b", "c")
    )
    llm = ScriptedLLMClient(tool_call_response(*calls), final_response())

    ToolLoop(llm, build_executor(*tools)).run(make_request(tools))

    assert events == [
        "a:start",
        "a:end",
        "b:start",
        "b:end",
        "c:start",
        "c:end",
    ]


def test_all_results_share_one_follow_up_request():
    calls = (tool_call("call-1"), tool_call("call-2"), tool_call("call-3"))
    llm = ScriptedLLMClient(tool_call_response(*calls), final_response())
    tool = calculator_tool()

    ToolLoop(llm, build_executor(tool)).run(make_request((tool,)))

    follow_up = llm.requests[1]
    assert [m.role for m in follow_up.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "tool",
    ]


def test_each_tool_message_matches_its_call_id():
    calls = (tool_call("call-1"), tool_call("call-2"), tool_call("call-3"))
    llm = ScriptedLLMClient(tool_call_response(*calls), final_response())
    tool = calculator_tool()

    ToolLoop(llm, build_executor(tool)).run(make_request((tool,)))

    tool_ids = [
        m.tool_call_id for m in llm.requests[1].messages if m.role == "tool"
    ]
    assert tool_ids == ["call-1", "call-2", "call-3"]


def test_executor_call_count_matches_the_calls():
    calls = (tool_call("call-1"), tool_call("call-2"), tool_call("call-3"))
    llm = ScriptedLLMClient(tool_call_response(*calls), final_response())
    executor = build_executor(calculator_tool())

    ToolLoop(llm, executor).run(make_request((calculator_tool(),)))

    assert len(executor.calls) == 3


# ------------------------------------------------------------- multiple rounds


def test_multi_round_tool_calls_are_supported():
    first = tool_call("call-1")
    second = tool_call("call-2")
    expected = final_response("done")
    llm = ScriptedLLMClient(
        tool_call_response(first), tool_call_response(second), expected
    )
    executor = build_executor(calculator_tool())
    tool = calculator_tool()

    response = ToolLoop(llm, executor).run(make_request((tool,)))

    assert response is expected
    assert llm.calls == 3
    assert executor.calls == [first, second]


def test_history_accumulates_every_round():
    first = tool_call("call-1")
    second = tool_call("call-2")
    llm = ScriptedLLMClient(
        tool_call_response(first), tool_call_response(second), final_response()
    )
    tool = calculator_tool()

    ToolLoop(llm, build_executor(tool)).run(
        make_request((tool,), messages=(Message.user("go"),))
    )

    assert [m.role for m in llm.requests[1].messages] == [
        "user",
        "assistant",
        "tool",
    ]
    assert [m.role for m in llm.requests[2].messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert llm.requests[2].messages[3].tool_calls == (second,)
    assert llm.requests[2].messages[4].tool_call_id == "call-2"


def test_history_keeps_every_assistant_and_tool_message():
    llm = ScriptedLLMClient(
        tool_call_response(tool_call("call-1")),
        tool_call_response(tool_call("call-2")),
        final_response(),
    )
    tool = calculator_tool()

    ToolLoop(llm, build_executor(tool)).run(make_request((tool,)))

    messages = llm.requests[2].messages
    assert sum(1 for m in messages if m.role == "assistant") == 2
    assert sum(1 for m in messages if m.role == "tool") == 2


def test_tools_are_preserved_on_every_round():
    llm = ScriptedLLMClient(
        tool_call_response(tool_call("call-1")),
        tool_call_response(tool_call("call-2")),
        final_response(),
    )
    tool = calculator_tool()

    ToolLoop(llm, build_executor(tool)).run(make_request((tool,)))

    assert llm.calls == 3
    for served in llm.requests:
        assert served.tools == (tool,)


# --------------------------------------------------------------- tool failures


def test_tool_failure_is_recovered_and_the_loop_continues():
    tool = failing_tool()
    call = tool_call(name="failing", arguments={})
    expected = final_response("recovered")
    llm = ScriptedLLMClient(tool_call_response(call), expected)

    response = ToolLoop(llm, build_executor(tool)).run(make_request((tool,)))

    assert response is expected
    assert llm.calls == 2


def test_tool_failure_becomes_an_error_tool_message():
    tool = failing_tool("failing", "boom")
    call = tool_call(name="failing", arguments={})
    llm = ScriptedLLMClient(tool_call_response(call), final_response())

    ToolLoop(llm, build_executor(tool)).run(make_request((tool,)))

    tool_message = llm.requests[1].messages[2]
    assert tool_message.role == "tool"
    assert tool_message.tool_call_id == call.id
    assert "failing" in tool_message.content
    assert "boom" in tool_message.content


def test_tool_failure_produces_an_error_tool_result():
    tool = failing_tool("failing", "boom")
    call = tool_call(name="failing", arguments={})
    llm = ScriptedLLMClient(tool_call_response(call), final_response())
    loop = ResultCapturingToolLoop(llm, build_executor(tool))

    loop.run(make_request((tool,)))

    assert len(loop.results) == 1
    result = loop.results[0]
    assert isinstance(result, ToolResult)
    assert result.is_error is True
    assert result.tool_call_id == "call-1"
    assert result.name == "failing"
    assert "boom" in result.output


def test_tool_failure_message_leaks_no_internals():
    tool = failing_tool("failing", "boom")
    call = tool_call(name="failing", arguments={})
    llm = ScriptedLLMClient(tool_call_response(call), final_response())
    loop = ResultCapturingToolLoop(llm, build_executor(tool))

    loop.run(make_request((tool,)))

    output = loop.results[0].output
    assert "Traceback" not in output
    assert 'File "' not in output
    assert ".py" not in output
    assert "sk-" not in output


def test_tool_failure_does_not_raise_or_add_rounds():
    tool = failing_tool()
    call = tool_call(name="failing", arguments={})
    llm = ScriptedLLMClient(tool_call_response(call), final_response())

    ToolLoop(llm, build_executor(tool)).run(make_request((tool,)))

    assert llm.calls == 2


def test_tool_failure_is_not_a_tool_loop_error():
    tool = failing_tool()
    call = tool_call(name="failing", arguments={})
    llm = ScriptedLLMClient(tool_call_response(call), final_response())
    loop = ToolLoop(llm, build_executor(tool))

    loop.run(make_request((tool,)))  # must not raise

    assert llm.calls == 2

# ------------------------------------------------------- unknown tool contract


def test_unknown_tool_propagates_as_tool_registry_error():
    call = tool_call(name="missing", arguments={})
    llm = ScriptedLLMClient(tool_call_response(call), final_response())
    executor = build_executor(calculator_tool())

    with pytest.raises(ToolRegistryError, match="unknown tool 'missing'"):
        ToolLoop(llm, executor).run(make_request((calculator_tool(),)))

    assert llm.calls == 1
    assert executor.calls == [call]


def test_unknown_tool_is_not_turned_into_a_result():
    call = tool_call(name="missing", arguments={})
    llm = ScriptedLLMClient(tool_call_response(call), final_response())
    loop = ResultCapturingToolLoop(llm, build_executor(calculator_tool()))

    with pytest.raises(ToolRegistryError):
        loop.run(make_request((calculator_tool(),)))

    assert loop.results == []


def test_unknown_tool_stops_before_another_llm_round():
    call = tool_call(name="missing", arguments={})
    llm = ScriptedLLMClient(tool_call_response(call), final_response())

    with pytest.raises(ToolRegistryError):
        ToolLoop(llm, build_executor(calculator_tool())).run(
            make_request((calculator_tool(),))
        )

    assert len(llm.requests) == 1


def test_unknown_tool_is_not_silently_ignored():
    call = tool_call(name="missing", arguments={})
    llm = ScriptedLLMClient(tool_call_response(call), final_response())

    with pytest.raises(ToolRegistryError):
        ToolLoop(llm, build_executor(calculator_tool())).run(
            make_request((calculator_tool(),))
        )

    assert llm.calls == 1


# -------------------------------------------------------------------- LLMError


def test_llm_error_on_the_first_round_propagates():
    llm = ScriptedLLMClient(LLMError("provider down"))

    with pytest.raises(LLMError, match="provider down"):
        ToolLoop(llm, build_executor(calculator_tool())).run(
            make_request((calculator_tool(),))
        )

    assert llm.calls == 1


def test_llm_error_is_not_retried_on_the_first_round():
    llm = ScriptedLLMClient(LLMError("provider down"), final_response())

    with pytest.raises(LLMError):
        ToolLoop(llm, build_executor(calculator_tool())).run(
            make_request((calculator_tool(),))
        )

    assert llm.calls == 1


def test_llm_error_on_a_later_round_propagates():
    call = tool_call()
    llm = ScriptedLLMClient(tool_call_response(call), LLMError("provider down"))
    executor = build_executor(calculator_tool())

    with pytest.raises(LLMError, match="provider down"):
        ToolLoop(llm, executor).run(make_request((calculator_tool(),)))

    assert llm.calls == 2
    assert executor.calls == [call]


def test_llm_error_is_not_retried_on_a_later_round():
    call = tool_call()
    llm = ScriptedLLMClient(
        tool_call_response(call), LLMError("provider down"), final_response()
    )

    with pytest.raises(LLMError):
        ToolLoop(llm, build_executor(calculator_tool())).run(
            make_request((calculator_tool(),))
        )

    assert llm.calls == 2


def test_llm_error_never_becomes_a_tool_result():
    call = tool_call()
    llm = ScriptedLLMClient(tool_call_response(call), LLMError("provider down"))
    loop = ResultCapturingToolLoop(llm, build_executor(calculator_tool()))

    with pytest.raises(LLMError):
        loop.run(make_request((calculator_tool(),)))

    assert len(loop.results) == 1
    assert loop.results[0].is_error is False


# ------------------------------------------------------------------ max_rounds


def test_max_rounds_one_allows_a_final_answer():
    llm = ScriptedLLMClient(final_response("done"))

    response = ToolLoop(llm, build_executor(calculator_tool()), max_rounds=1).run(
        make_request((calculator_tool(),))
    )

    assert response.content == "done"
    assert llm.calls == 1


def test_max_rounds_one_rejects_a_tool_call():
    call = tool_call()
    llm = ScriptedLLMClient(tool_call_response(call), final_response())
    executor = build_executor(calculator_tool())

    with pytest.raises(ToolLoopError, match="max_rounds=1"):
        ToolLoop(llm, executor, max_rounds=1).run(make_request((calculator_tool(),)))

    assert llm.calls == 1
    assert executor.calls == []


def test_max_rounds_two_allows_one_tool_round():
    call = tool_call()
    expected = final_response("done")
    llm = ScriptedLLMClient(tool_call_response(call), expected)
    executor = build_executor(calculator_tool())

    response = ToolLoop(llm, executor, max_rounds=2).run(
        make_request((calculator_tool(),))
    )

    assert response is expected
    assert llm.calls == 2
    assert executor.calls == [call]


def test_max_rounds_two_rejects_a_second_tool_round():
    llm = ScriptedLLMClient(
        tool_call_response(tool_call("call-1")),
        tool_call_response(tool_call("call-2")),
        final_response(),
    )
    executor = build_executor(calculator_tool())

    with pytest.raises(ToolLoopError, match="max_rounds=2"):
        ToolLoop(llm, executor, max_rounds=2).run(make_request((calculator_tool(),)))

    assert llm.calls == 2
    assert len(executor.calls) == 1


def test_continuous_tool_calls_terminate():
    script = [tool_call_response(tool_call(f"call-{i}")) for i in range(20)]
    llm = ScriptedLLMClient(*script)
    executor = build_executor(calculator_tool())

    with pytest.raises(ToolLoopError):
        ToolLoop(llm, executor, max_rounds=4).run(make_request((calculator_tool(),)))

    assert llm.calls == 4


@pytest.mark.parametrize("max_rounds", [1, 2, 3, 5, 8])
def test_llm_call_count_never_exceeds_max_rounds(max_rounds):
    script = [tool_call_response(tool_call(f"call-{i}")) for i in range(max_rounds + 3)]
    llm = ScriptedLLMClient(*script)

    with pytest.raises(ToolLoopError):
        ToolLoop(llm, build_executor(calculator_tool()), max_rounds=max_rounds).run(
            make_request((calculator_tool(),))
        )

    assert llm.calls == max_rounds


def test_default_max_rounds_is_eight():
    script = [tool_call_response(tool_call(f"call-{i}")) for i in range(12)]
    llm = ScriptedLLMClient(*script)
    executor = build_executor(calculator_tool())

    with pytest.raises(ToolLoopError, match="max_rounds=8"):
        ToolLoop(llm, executor).run(make_request((calculator_tool(),)))

    assert llm.calls == 8

# ------------------------------------------------------- request immutability


def test_original_request_is_never_mutated():
    tool = calculator_tool()
    messages = (Message.user("add 1 and 2"),)
    request = make_request((tool,), messages=messages)
    llm = ScriptedLLMClient(tool_call_response(tool_call()), final_response())

    ToolLoop(llm, build_executor(tool)).run(request)

    assert request.messages is messages
    assert request.messages == (Message.user("add 1 and 2"),)
    assert len(request.messages) == 1


def test_original_tools_tuple_is_unchanged():
    tool = calculator_tool()
    tools = (tool,)
    request = make_request(tools)
    llm = ScriptedLLMClient(tool_call_response(tool_call()), final_response())

    ToolLoop(llm, build_executor(tool)).run(request)

    assert request.tools is tools


def test_follow_up_requests_use_a_fresh_message_tuple():
    tool = calculator_tool()
    request = make_request((tool,))
    llm = ScriptedLLMClient(tool_call_response(tool_call()), final_response())

    ToolLoop(llm, build_executor(tool)).run(request)

    follow_up = llm.requests[1]
    assert isinstance(follow_up.messages, tuple)
    assert follow_up.messages is not request.messages
    assert request.messages == (Message.user("go"),)


def test_follow_up_request_preserves_every_other_field():
    tool = calculator_tool()
    request = make_request(
        (tool,),
        messages=(Message.user("go"),),
        temperature=0.7,
        max_tokens=256,
        stop=("\n",),
        json_mode=True,
        metadata={"k": "v"},
    )
    llm = ScriptedLLMClient(tool_call_response(tool_call()), final_response())

    ToolLoop(llm, build_executor(tool)).run(request)

    follow_up = llm.requests[1]
    assert follow_up.temperature == 0.7
    assert follow_up.max_tokens == 256
    assert follow_up.stop == ("\n",)
    assert follow_up.json_mode is True
    assert follow_up.metadata == {"k": "v"}
    assert follow_up.tools == (tool,)


# ------------------------------------------------------ request.tools contract


def test_tool_call_without_declared_tools_is_a_contract_violation():
    call = tool_call()
    llm = ScriptedLLMClient(tool_call_response(call))
    executor = build_executor(calculator_tool())

    with pytest.raises(ToolLoopError, match="declared none"):
        ToolLoop(llm, executor).run(make_request())

    assert llm.calls == 1
    assert executor.calls == []


def test_registered_tools_are_not_discovered_without_declaring_them():
    call = tool_call(name="calculator", arguments={})
    llm = ScriptedLLMClient(tool_call_response(call))
    executor = build_executor(calculator_tool())

    with pytest.raises(ToolLoopError, match="declared none"):
        ToolLoop(llm, executor).run(make_request())

    assert executor.calls == []


def test_undeclared_tools_do_not_reach_the_model():
    call = tool_call()
    llm = ScriptedLLMClient(tool_call_response(call))

    with pytest.raises(ToolLoopError):
        ToolLoop(llm, build_executor(calculator_tool())).run(make_request())

    assert llm.requests[0].tools == ()


# ------------------------------------------------------------- message history


def test_assistant_message_precedes_tool_messages():
    calls = (tool_call("call-1"), tool_call("call-2"))
    llm = ScriptedLLMClient(tool_call_response(*calls), final_response())
    tool = calculator_tool()

    ToolLoop(llm, build_executor(tool)).run(make_request((tool,)))

    roles = [m.role for m in llm.requests[1].messages]
    assert roles.index("assistant") < roles.index("tool")


def test_multiple_tool_messages_keep_their_order():
    calls = (tool_call("call-1"), tool_call("call-2"), tool_call("call-3"))
    llm = ScriptedLLMClient(tool_call_response(*calls), final_response())
    tool = calculator_tool()

    ToolLoop(llm, build_executor(tool)).run(make_request((tool,)))

    tool_ids = [
        m.tool_call_id for m in llm.requests[1].messages if m.role == "tool"
    ]
    assert tool_ids == ["call-1", "call-2", "call-3"]


def test_assistant_content_is_kept_on_the_tool_call_message():
    call = tool_call()
    llm = ScriptedLLMClient(
        tool_call_response(call, content="let me compute"), final_response()
    )
    tool = calculator_tool()

    ToolLoop(llm, build_executor(tool)).run(make_request((tool,)))

    assert llm.requests[1].messages[1].content == "let me compute"


def test_history_never_loses_earlier_exchanges():
    llm = ScriptedLLMClient(
        tool_call_response(tool_call("call-1")),
        tool_call_response(tool_call("call-2")),
        final_response(),
    )
    tool = calculator_tool()

    ToolLoop(llm, build_executor(tool)).run(make_request((tool,)))

    second_round_history = list(llm.requests[1].messages)
    third_round_history = list(llm.requests[2].messages)
    assert third_round_history[: len(second_round_history)] == second_round_history


# -------------------------------------------------------------------- isolation


def _imports_of_path(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))
    return imported


def _imports_of(module_name: str) -> list[str]:
    module = importlib.import_module(module_name)
    return _imports_of_path(pathlib.Path(module.__file__))


def test_tool_loop_only_depends_on_neutral_contracts():
    imported = _imports_of("orchestrator_worker.tool_loop")

    assert imported == ["__future__", "dataclasses", ".llm.base", ".tools"]


@pytest.mark.parametrize(
    "token",
    [
        "openai",
        "deepseek",
        "anthropic",
        "workers",
        "planner",
        "evaluators",
        "aggregators",
        "orchestrator",
    ],
)
def test_tool_loop_never_imports_provider_or_agent_layers(token):
    imported = _imports_of("orchestrator_worker.tool_loop")

    assert not any(token in name for name in imported)


@pytest.mark.parametrize(
    "token", ["asyncio", "concurrent", "threading", "multiprocessing"]
)
def test_tool_loop_uses_no_concurrency_imports(token):
    imported = _imports_of("orchestrator_worker.tool_loop")

    assert not any(token in name for name in imported)


@pytest.mark.parametrize(
    "module_name",
    [
        "orchestrator_worker.orchestrator",
        "orchestrator_worker.application",
        "orchestrator_worker.tools",
        "orchestrator_worker.retry",
        "orchestrator_worker.plan",
        "orchestrator_worker.router",
        "orchestrator_worker.context",
        "orchestrator_worker.execution",
        "orchestrator_worker.state",
        "orchestrator_worker.evaluation",
        "orchestrator_worker.llm.base",
        "orchestrator_worker.llm.deepseek",
        "orchestrator_worker.workers.base",
        "orchestrator_worker.workers.research",
        "orchestrator_worker.workers.coding",
        "orchestrator_worker.workers.registry",
        "orchestrator_worker.planner.base",
        "orchestrator_worker.planner.llm",
        "orchestrator_worker.evaluators.base",
        "orchestrator_worker.evaluators.llm",
        "orchestrator_worker.aggregators.base",
        "orchestrator_worker.aggregators.llm",
    ],
)
def test_no_production_module_imports_the_tool_loop(module_name):
    assert not any("tool_loop" in name for name in _imports_of(module_name))


def test_the_tool_loop_tests_never_import_a_provider_sdk():
    imported = _imports_of_path(pathlib.Path(__file__))

    assert not any(
        token in name
        for name in imported
        for token in ("openai", "deepseek", "anthropic")
    )


def test_tool_loop_returns_provider_neutral_types_only():
    llm = ScriptedLLMClient(final_response())

    response = ToolLoop(llm, build_executor(calculator_tool())).run(
        make_request((calculator_tool(),))
    )

    assert type(response) is LLMResponse
    assert response.tool_calls == ()