"""Phase 5C-2 tests: the provider-neutral LLM tool-call contract.

Everything here runs against a stub OpenAI-compatible client built from
``SimpleNamespace`` objects, so there is no network access and no ``openai``
import. Parsing/validation lives in the adapter; these tests pin both the wire
shape and the neutral objects.
"""

from __future__ import annotations

import ast
import importlib
import json
import pathlib
from types import SimpleNamespace

import pytest

from orchestrator_worker.llm import LLMError, LLMRequest, LLMResponse, Message
from orchestrator_worker.llm.deepseek import DeepSeekClient
from orchestrator_worker.tools import Tool, ToolCall

_MISSING = object()

CALCULATOR_PARAMETERS = {
    "type": "object",
    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
    "required": ["a", "b"],
}


def calculator() -> Tool:
    return Tool(
        name="calculator",
        description="Perform a simple arithmetic operation.",
        parameters=CALCULATOR_PARAMETERS,
        handler=lambda args: str(args["a"] + args["b"]),
    )


def echo() -> Tool:
    return Tool(
        name="echo",
        description="Echo the given text back.",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        handler=lambda args: args.get("text", ""),
    )


def make_tool_call(
    call_id: str = "call-1",
    name: str = "calculator",
    arguments: dict | None = None,
) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=name,
        arguments={"a": 1, "b": 2} if arguments is None else arguments,
    )


# ------------------------------------------------------------------ fake SDK


def sdk_function(name=_MISSING, arguments=_MISSING) -> SimpleNamespace:
    return SimpleNamespace(
        name="calculator" if name is _MISSING else name,
        arguments='{"a": 1, "b": 2}' if arguments is _MISSING else arguments,
    )


def sdk_tool_call(call_id=_MISSING, *, call_type=_MISSING, function=_MISSING):
    return SimpleNamespace(
        id="call-1" if call_id is _MISSING else call_id,
        type="function" if call_type is _MISSING else call_type,
        function=sdk_function() if function is _MISSING else function,
    )


def sdk_message(content=None, tool_calls=None) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def sdk_response(message, *, model="deepseek-chat", finish_reason="tool_calls"):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7, total_tokens=12),
    )


class StubCompletions:
    """Mimics client.chat.completions and records the outgoing payload."""

    def __init__(self, response: object) -> None:
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def send(request: LLMRequest, response=None) -> tuple[dict, LLMResponse]:
    """Send ``request`` through the adapter; return (payload, response)."""
    if response is None:
        response = sdk_response(sdk_message(content="hi"), finish_reason="stop")
    completions = StubCompletions(response)
    stub = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = DeepSeekClient(client=stub).complete(request)
    return completions.calls[0], result


def parse_one(raw_call) -> LLMResponse:
    """Run one provider tool call through the adapter."""
    _, result = send(
        LLMRequest(messages=[Message.user("hi")]),
        sdk_response(sdk_message(tool_calls=[raw_call])),
    )
    return result


# ------------------------------------------------------------- A. LLMRequest


def test_llm_request_defaults_to_no_tools():
    assert LLMRequest(messages=[Message.user("hi")]).tools == ()


def test_llm_request_accepts_a_tuple_of_tool_definitions():
    tool = calculator()

    assert LLMRequest(messages=[Message.user("hi")], tools=(tool,)).tools == (tool,)


def test_llm_request_rejects_non_tool_entries():
    with pytest.raises(LLMError, match=r"tools\[0\] must be a Tool"):
        LLMRequest(messages=[Message.user("hi")], tools=("calculator",))


def test_llm_request_rejects_a_non_tuple_tools_value():
    with pytest.raises(LLMError, match="tools must be a tuple"):
        LLMRequest(messages=[Message.user("hi")], tools=[calculator()])


def test_llm_request_keeps_its_original_construction():
    request = LLMRequest(
        messages=[Message.system("s"), Message.user("u")],
        temperature=0.2,
        max_tokens=64,
        stop=["x"],
        json_mode=True,
        metadata={"k": "v"},
    )

    assert request.temperature == 0.2
    assert request.max_tokens == 64
    assert request.stop == ["x"]
    assert request.json_mode is True
    assert request.metadata == {"k": "v"}
    assert request.tools == ()


# ------------------------------------------------------------ B. LLMResponse


def test_llm_response_defaults_to_no_tool_calls():
    assert LLMResponse(content="hi", model="m").tool_calls == ()


def test_llm_response_accepts_tool_calls():
    call = make_tool_call()

    assert LLMResponse(content="", model="m", tool_calls=(call,)).tool_calls == (call,)


def test_llm_response_rejects_non_tool_call_entries():
    with pytest.raises(LLMError, match=r"tool_calls\[0\] must be a ToolCall"):
        LLMResponse(content="", model="m", tool_calls=("call-1",))


def test_llm_response_rejects_a_non_tuple_tool_calls_value():
    with pytest.raises(LLMError, match="tool_calls must be a tuple"):
        LLMResponse(content="", model="m", tool_calls=[make_tool_call()])


def test_llm_response_keeps_its_original_construction():
    response = LLMResponse(content="hi", model="m", finish_reason="stop")

    assert response.content == "hi"
    assert response.model == "m"
    assert response.finish_reason == "stop"
    assert response.tool_calls == ()


# ---------------------------------------------------------------- C. Message


def test_plain_messages_are_unchanged():
    assert Message.user("hi") == Message(role="user", content="hi")
    assert Message.user("hi").tool_calls == ()
    assert Message.user("hi").tool_call_id is None
    assert Message.user("hi").to_dict() == {"role": "user", "content": "hi"}
    assert Message.system("s").to_dict() == {"role": "system", "content": "s"}
    assert Message.assistant("a").to_dict() == {"role": "assistant", "content": "a"}


def test_assistant_message_can_carry_tool_calls():
    call = make_tool_call()

    message = Message(role="assistant", content="", tool_calls=(call,))

    assert message.tool_calls == (call,)
    assert message.tool_call_id is None


def test_tool_message_carries_a_tool_call_id():
    message = Message.tool("42", "call-1")

    assert message.role == "tool"
    assert message.tool_call_id == "call-1"
    assert message.to_dict() == {
        "role": "tool",
        "content": "42",
        "tool_call_id": "call-1",
    }


def test_tool_message_requires_a_tool_call_id():
    with pytest.raises(LLMError, match="must set tool_call_id"):
        Message(role="tool", content="42")


@pytest.mark.parametrize("value", ["", "   "])
def test_tool_message_rejects_an_empty_tool_call_id(value):
    with pytest.raises(LLMError, match="tool_call_id must be a non-empty string"):
        Message.tool("42", value)


@pytest.mark.parametrize("role", ["system", "user", "tool"])
def test_non_assistant_messages_reject_tool_calls(role):
    with pytest.raises(LLMError, match="only allowed on an assistant message"):
        Message(role=role, content="x", tool_calls=(make_tool_call(),))


@pytest.mark.parametrize("role", ["system", "user", "assistant"])
def test_non_tool_messages_reject_a_tool_call_id(role):
    with pytest.raises(LLMError, match="only allowed on a tool message"):
        Message(role=role, content="x", tool_call_id="call-1")


def test_message_rejects_unknown_roles():
    with pytest.raises(LLMError, match="Message.role must be one of"):
        Message(role="developer", content="x")


def test_message_rejects_non_tool_call_entries():
    with pytest.raises(LLMError, match=r"tool_calls\[0\] must be a ToolCall"):
        Message(role="assistant", content="", tool_calls=("call-1",))


def test_message_rejects_a_non_tuple_tool_calls_value():
    with pytest.raises(LLMError, match="tool_calls must be a tuple"):
        Message(role="assistant", content="", tool_calls=[make_tool_call()])


def test_message_content_must_be_a_string():
    with pytest.raises(LLMError, match="content must be a string"):
        Message(role="user", content=None)


# ------------------------------------------------- D. request serialization


def test_tools_are_not_sent_when_there_are_none():
    payload, _ = send(LLMRequest(messages=[Message.user("hi")]))

    assert "tools" not in payload


def test_one_tool_is_serialized_as_a_function_definition():
    payload, _ = send(
        LLMRequest(messages=[Message.user("hi")], tools=(calculator(),))
    )

    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform a simple arithmetic operation.",
                "parameters": CALCULATOR_PARAMETERS,
            },
        }
    ]


def test_multiple_tools_keep_their_order():
    payload, _ = send(
        LLMRequest(messages=[Message.user("hi")], tools=(calculator(), echo()))
    )

    assert [entry["function"]["name"] for entry in payload["tools"]] == [
        "calculator",
        "echo",
    ]


def test_tool_parameters_are_passed_through_unchanged():
    parameters = {"type": "object", "properties": {"x": {"type": "string"}}}
    tool = Tool(
        name="t", description="d", parameters=parameters, handler=lambda args: ""
    )

    payload, _ = send(LLMRequest(messages=[Message.user("hi")], tools=(tool,)))

    assert payload["tools"][0]["function"]["parameters"] is parameters


def test_the_handler_never_reaches_the_payload():
    payload, _ = send(
        LLMRequest(messages=[Message.user("hi")], tools=(calculator(),))
    )

    definition = payload["tools"][0]["function"]
    assert set(definition) == {"name", "description", "parameters"}
    assert "handler" not in json.dumps(payload)


# -------------------------------------------------- E. response parsing


def test_one_tool_call_is_parsed_into_a_tool_call():
    result = parse_one(sdk_tool_call())

    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert isinstance(call, ToolCall)
    assert call.id == "call-1"
    assert call.name == "calculator"
    assert call.arguments == {"a": 1, "b": 2}


def test_multiple_tool_calls_are_parsed_in_order():
    response = sdk_response(
        sdk_message(
            tool_calls=[
                sdk_tool_call("call-1", function=sdk_function("calculator", '{"a": 1}')),
                sdk_tool_call("call-2", function=sdk_function("echo", '{"text": "hi"}')),
            ]
        )
    )

    _, result = send(LLMRequest(messages=[Message.user("hi")]), response)

    assert [(call.id, call.name, call.arguments) for call in result.tool_calls] == [
        ("call-1", "calculator", {"a": 1}),
        ("call-2", "echo", {"text": "hi"}),
    ]


def test_arguments_key_order_does_not_matter():
    result = parse_one(sdk_tool_call(function=sdk_function(arguments='{"b": 2, "a": 1}')))

    assert result.tool_calls[0].arguments == {"a": 1, "b": 2}


def test_content_is_normalized_when_the_model_only_returns_tool_calls():
    result = parse_one(sdk_tool_call())

    assert result.content == ""


def test_no_tool_calls_means_an_empty_tuple():
    _, result = send(
        LLMRequest(messages=[Message.user("hi")]),
        sdk_response(sdk_message(content="hi"), finish_reason="stop"),
    )

    assert result.tool_calls == ()


@pytest.mark.parametrize("raw_arguments", ["{not json", "{'a': 1}", "", "   "])
def test_invalid_argument_strings_are_rejected(raw_arguments):
    with pytest.raises(LLMError, match="tool call #0"):
        parse_one(sdk_tool_call(function=sdk_function(arguments=raw_arguments)))


@pytest.mark.parametrize(
    "raw_arguments", ["[]", '"text"', "42", "3.5", "true", "false", "null"]
)
def test_non_object_arguments_are_rejected(raw_arguments):
    with pytest.raises(LLMError, match="must be a JSON object"):
        parse_one(sdk_tool_call(function=sdk_function(arguments=raw_arguments)))


@pytest.mark.parametrize("call_id", [None, "", "   "])
def test_missing_or_empty_id_is_rejected(call_id):
    with pytest.raises(LLMError, match="missing a valid id"):
        parse_one(sdk_tool_call(call_id=call_id))


def test_missing_function_is_rejected():
    with pytest.raises(LLMError, match="missing its function"):
        parse_one(sdk_tool_call(function=None))


@pytest.mark.parametrize("name", [None, "", "   "])
def test_missing_or_empty_name_is_rejected(name):
    with pytest.raises(LLMError, match="missing a valid name"):
        parse_one(sdk_tool_call(function=sdk_function(name=name)))


@pytest.mark.parametrize("arguments", [None, "", "   "])
def test_missing_arguments_are_rejected(arguments):
    with pytest.raises(LLMError, match="missing its arguments"):
        parse_one(sdk_tool_call(function=sdk_function(arguments=arguments)))


@pytest.mark.parametrize("call_type", ["code_interpreter", "retrieval", "", None])
def test_unsupported_tool_types_are_rejected(call_type):
    with pytest.raises(LLMError, match="unsupported type"):
        parse_one(sdk_tool_call(call_type=call_type))


def test_no_sdk_objects_leak_into_the_response():
    result = parse_one(sdk_tool_call())

    for call in result.tool_calls:
        assert type(call) is ToolCall
        assert type(call.arguments) is dict
        for value in call.arguments.values():
            assert not isinstance(value, SimpleNamespace)


# ------------------------------------- F/G. tool-call message serialization


def test_assistant_tool_call_message_is_serialized_for_the_provider():
    message = Message(role="assistant", content="", tool_calls=(make_tool_call(),))

    payload, _ = send(LLMRequest(messages=[message]))

    assert payload["messages"] == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"a": 1, "b": 2}',
                    },
                }
            ],
        }
    ]


def test_arguments_are_encoded_as_a_json_string():
    message = Message(
        role="assistant", content="", tool_calls=(make_tool_call(arguments={"b": 2, "a": 1}),)
    )

    payload, _ = send(LLMRequest(messages=[message]))

    encoded = payload["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(encoded, str)
    assert json.loads(encoded) == {"a": 1, "b": 2}


def test_tool_result_message_is_serialized_for_the_provider():
    payload, _ = send(LLMRequest(messages=[Message.tool("42", "call-1")]))

    assert payload["messages"] == [
        {"role": "tool", "content": "42", "tool_call_id": "call-1"}
    ]


def test_a_full_assistant_then_tool_exchange_serializes():
    payload, _ = send(
        LLMRequest(
            messages=[
                Message.user("add 1 and 2"),
                Message(role="assistant", content="", tool_calls=(make_tool_call(),)),
                Message.tool("3", "call-1"),
            ]
        )
    )

    assert [message["role"] for message in payload["messages"]] == [
        "user",
        "assistant",
        "tool",
    ]
    assert payload["messages"][1]["tool_calls"][0]["function"]["name"] == "calculator"
    assert payload["messages"][2]["tool_call_id"] == "call-1"


# ------------------------------------------------------------- H. isolation


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


def test_base_module_consumes_the_tool_contract_without_a_provider():
    imported = _imports_of("orchestrator_worker.llm.base")

    assert "..tools" in imported
    forbidden = (
        "openai",
        "deepseek",
        "anthropic",
        "workers",
        "planner",
        "evaluators",
        "aggregators",
        "orchestrator",
    )
    assert not any(token in name for name in imported for token in forbidden)


def test_deepseek_module_does_not_import_agent_layers():
    imported = _imports_of("orchestrator_worker.llm.deepseek")

    assert "..tools" in imported
    forbidden = ("workers", "planner", "evaluators", "aggregators", "orchestrator")
    assert not any(token in name for name in imported for token in forbidden)


def test_adapter_tests_never_import_the_openai_sdk():
    imported = _imports_of_path(pathlib.Path(__file__))

    assert not any("openai" in name for name in imported)