"""Phase 5C-1 tests: the provider-neutral tool foundation."""

from __future__ import annotations

import ast
import dataclasses
import pathlib

import pytest

from orchestrator_worker.tools import (
    Tool,
    ToolCall,
    ToolError,
    ToolExecutionError,
    ToolExecutor,
    ToolRegistry,
    ToolRegistryError,
    ToolResult,
)

CALCULATOR_PARAMETERS = {
    "type": "object",
    "properties": {
        "a": {"type": "number"},
        "b": {"type": "number"},
    },
    "required": ["a", "b"],
}


def calculator_tool(handler=None) -> Tool:
    return Tool(
        name="calculator",
        description="Perform a simple arithmetic operation.",
        parameters=CALCULATOR_PARAMETERS,
        handler=handler if handler is not None else (lambda args: str(args["a"] + args["b"])),
    )


def build_registry(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def call_for(
    name: str = "calculator",
    arguments: dict | None = None,
    call_id: str = "call-1",
) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=name,
        arguments={"a": 1, "b": 2} if arguments is None else arguments,
    )


# --------------------------------------------------------------------------- Tool


def test_tool_creates_with_valid_fields():
    tool = calculator_tool()

    assert tool.name == "calculator"
    assert tool.description == "Perform a simple arithmetic operation."
    assert tool.parameters == CALCULATOR_PARAMETERS
    assert callable(tool.handler)


@pytest.mark.parametrize("value", ["", "   "])
def test_tool_rejects_empty_name(value):
    with pytest.raises(ToolError, match="Tool.name must be a non-empty string"):
        Tool(
            name=value,
            description="d",
            parameters={},
            handler=lambda args: None,
        )


@pytest.mark.parametrize("value", ["", "   "])
def test_tool_rejects_empty_description(value):
    with pytest.raises(ToolError, match="Tool.description must be a non-empty string"):
        Tool(name="n", description=value, parameters={}, handler=lambda args: None)


@pytest.mark.parametrize("value", ["not a mapping", 42, ["a"], None])
def test_tool_rejects_non_mapping_parameters(value):
    with pytest.raises(ToolError, match="Tool.parameters must be a mapping"):
        Tool(name="n", description="d", parameters=value, handler=lambda args: None)


@pytest.mark.parametrize("value", ["nope", 42, None, {}])
def test_tool_rejects_non_callable_handler(value):
    with pytest.raises(ToolError, match="Tool.handler must be callable"):
        Tool(name="n", description="d", parameters={}, handler=value)


def test_tool_is_frozen():
    tool = calculator_tool()

    with pytest.raises(dataclasses.FrozenInstanceError):
        tool.name = "other"


def test_tool_has_exactly_the_expected_fields():
    assert [field.name for field in dataclasses.fields(Tool)] == [
        "name",
        "description",
        "parameters",
        "handler",
    ]


# ----------------------------------------------------------------------- ToolCall


def test_tool_call_creates_with_valid_fields():
    call = ToolCall(id="call-1", name="calculator", arguments={"a": 1, "b": 2})

    assert call.id == "call-1"
    assert call.name == "calculator"
    assert call.arguments == {"a": 1, "b": 2}


@pytest.mark.parametrize("value", ["", "   "])
def test_tool_call_rejects_empty_id(value):
    with pytest.raises(ToolError, match="ToolCall.id must be a non-empty string"):
        ToolCall(id=value, name="calculator", arguments={})


@pytest.mark.parametrize("value", ["", "   "])
def test_tool_call_rejects_empty_name(value):
    with pytest.raises(ToolError, match="ToolCall.name must be a non-empty string"):
        ToolCall(id="call-1", name=value, arguments={})


@pytest.mark.parametrize("value", ["{}", 42, ["a"], None])
def test_tool_call_rejects_non_dict_arguments(value):
    with pytest.raises(ToolError, match="ToolCall.arguments must be a dict"):
        ToolCall(id="call-1", name="calculator", arguments=value)


def test_tool_call_is_frozen():
    call = call_for()

    with pytest.raises(dataclasses.FrozenInstanceError):
        call.name = "other"


def test_tool_call_has_exactly_the_expected_fields():
    assert [field.name for field in dataclasses.fields(ToolCall)] == [
        "id",
        "name",
        "arguments",
    ]


# --------------------------------------------------------------------- ToolResult


def test_tool_result_success_fields():
    result = ToolResult(tool_call_id="call-1", name="calculator", output="42")

    assert result.tool_call_id == "call-1"
    assert result.name == "calculator"
    assert result.output == "42"
    assert result.is_error is False


def test_tool_result_error_flag():
    result = ToolResult(
        tool_call_id="call-1", name="calculator", output="boom", is_error=True
    )

    assert result.is_error is True


@pytest.mark.parametrize("value", [42, None, ["42"]])
def test_tool_result_rejects_non_string_output(value):
    with pytest.raises(ToolError, match="ToolResult.output must be a string"):
        ToolResult(tool_call_id="call-1", name="calculator", output=value)


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_tool_result_rejects_non_bool_is_error(value):
    with pytest.raises(ToolError, match="ToolResult.is_error must be a bool"):
        ToolResult(
            tool_call_id="call-1",
            name="calculator",
            output="42",
            is_error=value,
        )


def test_tool_result_is_frozen():
    result = ToolResult(tool_call_id="call-1", name="calculator", output="42")

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.output = "43"


def test_tool_result_has_exactly_the_expected_fields():
    assert [field.name for field in dataclasses.fields(ToolResult)] == [
        "tool_call_id",
        "name",
        "output",
        "is_error",
    ]


# ------------------------------------------------------------------- ToolRegistry


def test_registry_registers_a_tool():
    tool = calculator_tool()
    registry = build_registry(tool)

    assert registry.has("calculator") is True
    assert registry.get("calculator") is tool


def test_registry_has_is_false_for_unknown_names():
    registry = build_registry(calculator_tool())

    assert registry.has("missing") is False


def test_registry_get_raises_for_unknown_names():
    registry = build_registry(calculator_tool())

    with pytest.raises(ToolRegistryError, match="unknown tool 'missing'"):
        registry.get("missing")


def test_registry_rejects_duplicate_names():
    registry = build_registry(calculator_tool())

    with pytest.raises(ToolRegistryError, match="already registered"):
        registry.register(calculator_tool())


def test_registry_rejects_non_tool_registration():
    registry = ToolRegistry()

    with pytest.raises(ToolRegistryError, match="expects a Tool"):
        registry.register("calculator")


def test_registry_names_are_sorted():
    registry = build_registry(
        calculator_tool(),
        Tool(name="zeta", description="d", parameters={}, handler=lambda args: None),
        Tool(name="alpha", description="d", parameters={}, handler=lambda args: None),
    )

    assert registry.names() == ["alpha", "calculator", "zeta"]


def test_registry_tools_returns_a_tuple_sorted_by_name():
    registry = build_registry(
        calculator_tool(),
        Tool(name="alpha", description="d", parameters={}, handler=lambda args: None),
    )

    tools = registry.tools()

    assert isinstance(tools, tuple)
    assert [tool.name for tool in tools] == ["alpha", "calculator"]


def test_registry_cannot_be_mutated_through_returned_values():
    tool = calculator_tool()
    registry = build_registry(tool)

    names = registry.names()
    names.append("injected")
    names.clear()
    tools = registry.tools()

    assert registry.names() == ["calculator"]
    assert registry.tools() == tools
    assert registry.get("calculator") is tool


# ------------------------------------------------------------------- ToolExecutor


def test_executor_runs_the_handler_and_returns_a_result():
    executor = ToolExecutor(build_registry(calculator_tool()))

    result = executor.execute(call_for())

    assert isinstance(result, ToolResult)
    assert result.tool_call_id == "call-1"
    assert result.name == "calculator"
    assert result.output == "3"
    assert result.is_error is False


def test_executor_passes_the_parsed_arguments_through_unchanged():
    seen: list[dict] = []

    def handler(args):
        seen.append(args)
        return "ok"

    call = call_for(arguments={"a": 1, "b": 2})
    ToolExecutor(build_registry(calculator_tool(handler=handler))).execute(call)

    assert seen == [{"a": 1, "b": 2}]
    assert seen[0] is call.arguments


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (42, "42"),
        (3.5, "3.5"),
        (None, "None"),
        (True, "True"),
        ({"x": 1}, "{'x': 1}"),
        (["a", "b"], "['a', 'b']"),
    ],
)
def test_executor_stringifies_handler_output(raw, expected):
    tool = Tool(
        name="echo",
        description="Return whatever it was given.",
        parameters={},
        handler=lambda args, raw=raw: raw,
    )
    executor = ToolExecutor(build_registry(tool))

    result = executor.execute(ToolCall(id="call-1", name="echo", arguments={}))

    assert result.output == expected


def test_executor_propagates_unknown_tool_without_fallback():
    executor = ToolExecutor(build_registry(calculator_tool()))

    with pytest.raises(ToolRegistryError) as excinfo:
        executor.execute(call_for(name="missing", arguments={}))

    assert type(excinfo.value) is ToolRegistryError
    assert "registered tools: calculator" in str(excinfo.value)


def test_executor_wraps_handler_failures():
    error = ValueError("boom")

    def handler(args):
        raise error

    executor = ToolExecutor(build_registry(calculator_tool(handler=handler)))

    with pytest.raises(ToolExecutionError) as excinfo:
        executor.execute(call_for())

    assert "calculator" in str(excinfo.value)
    assert "boom" in str(excinfo.value)


def test_executor_preserves_the_original_exception_as_cause():
    error = ValueError("boom")

    def handler(args):
        raise error

    executor = ToolExecutor(build_registry(calculator_tool(handler=handler)))

    with pytest.raises(ToolExecutionError) as excinfo:
        executor.execute(call_for())

    assert excinfo.value.__cause__ is error


def test_executor_rejects_non_tool_call():
    executor = ToolExecutor(build_registry(calculator_tool()))

    with pytest.raises(ToolError, match="expects a ToolCall"):
        executor.execute("call-1")


def test_executor_rejects_non_registry():
    with pytest.raises(ToolError, match="expects a ToolRegistry"):
        ToolExecutor(object())


def test_executor_does_not_modify_the_tool_call():
    call = call_for()
    arguments_before = dict(call.arguments)

    ToolExecutor(build_registry(calculator_tool())).execute(call)

    assert call.id == "call-1"
    assert call.name == "calculator"
    assert call.arguments == arguments_before


def test_executor_does_not_modify_the_registry():
    tool = calculator_tool()
    registry = build_registry(tool)
    executor = ToolExecutor(registry)
    names_before = registry.names()

    executor.execute(call_for())

    assert registry.names() == names_before
    assert registry.get("calculator") is tool


# ------------------------------------------------------------------ error contract


def test_tool_error_hierarchy():
    assert issubclass(ToolError, RuntimeError)
    assert issubclass(ToolRegistryError, ToolError)
    assert issubclass(ToolExecutionError, ToolError)
    assert not issubclass(ToolExecutionError, ToolRegistryError)


def test_tools_module_is_provider_neutral():
    from orchestrator_worker import tools as tools_module

    tree = ast.parse(
        pathlib.Path(tools_module.__file__).read_text(encoding="utf-8")
    )
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))

    assert imported == ["__future__", "collections.abc", "dataclasses", "typing"]

    forbidden = (
        "openai",
        "deepseek",
        "anthropic",
        "llm",
        "workers",
        "planner",
        "evaluators",
        "aggregators",
        "orchestrator",
    )
    assert not any(token in name for name in imported for token in forbidden)