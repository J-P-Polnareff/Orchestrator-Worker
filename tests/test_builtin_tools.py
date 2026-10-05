"""Phase 5C-4B tests: the built-in calculator tool."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import pathlib

import pytest

from orchestrator_worker.builtin_tools import (
    CALCULATOR_DESCRIPTION,
    CALCULATOR_NAME,
    CALCULATOR_OPERATIONS,
    CALCULATOR_PARAMETERS,
    calculator_tool,
)
from orchestrator_worker.tools import (
    Tool,
    ToolCall,
    ToolExecutionError,
    ToolExecutor,
    ToolRegistry,
)

MODULE_NAME = "orchestrator_worker.builtin_tools"


def run_calculator(arguments: dict):
    return calculator_tool().handler(arguments)


def build_executor() -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(calculator_tool())
    return ToolExecutor(registry)


def module_source() -> str:
    module = importlib.import_module(MODULE_NAME)
    return pathlib.Path(module.__file__).read_text(encoding="utf-8")


def _imports_of_path(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))
    return imported


# ------------------------------------------------------------------- definition


def test_calculator_tool_is_a_tool():
    assert isinstance(calculator_tool(), Tool)


def test_calculator_name_and_description():
    tool = calculator_tool()

    assert CALCULATOR_NAME == "calculator"
    assert tool.name == "calculator"
    assert tool.description == CALCULATOR_DESCRIPTION
    assert tool.description.strip()


def test_calculator_parameters_schema():
    tool = calculator_tool()
    parameters = tool.parameters

    assert parameters["type"] == "object"
    assert set(parameters["properties"]) == {"a", "b", "operation"}
    assert parameters["properties"]["a"] == {"type": "number"}
    assert parameters["properties"]["b"] == {"type": "number"}
    assert parameters["properties"]["operation"]["type"] == "string"


def test_calculator_schema_is_strict():
    parameters = calculator_tool().parameters

    assert parameters["required"] == ["a", "b", "operation"]
    assert parameters["additionalProperties"] is False


def test_calculator_exposes_exactly_four_operations():
    assert CALCULATOR_OPERATIONS == ("add", "subtract", "multiply", "divide")

    enum = calculator_tool().parameters["properties"]["operation"]["enum"]

    assert enum == ["add", "subtract", "multiply", "divide"]


def test_calculator_parameters_are_the_shared_definition():
    assert calculator_tool().parameters == CALCULATOR_PARAMETERS


def test_calculator_tool_is_frozen():
    tool = calculator_tool()

    with pytest.raises(dataclasses.FrozenInstanceError):
        tool.name = "other"


def test_factory_returns_an_equal_but_fresh_definition():
    first = calculator_tool()
    second = calculator_tool()

    assert first == second
    assert first is not second


# ------------------------------------------------------------------ arithmetic


@pytest.mark.parametrize(
    ("a", "b", "operation", "expected"),
    [
        (2, 3, "add", 5),
        (2, 3, "subtract", -1),
        (17, 23, "multiply", 391),
        (-4, 2, "multiply", -8),
        (0, 5, "add", 5),
        (7, 2, "divide", 3.5),
        (2.5, 0.5, "add", 3.0),
    ],
)
def test_calculator_operations(a, b, operation, expected):
    assert run_calculator({"a": a, "b": b, "operation": operation}) == expected


def test_integer_arithmetic_stays_integral():
    result = run_calculator({"a": 17, "b": 23, "operation": "multiply"})

    assert result == 391
    assert isinstance(result, int)


def test_divide_by_zero_raises_value_error():
    with pytest.raises(ValueError, match="divide by zero"):
        run_calculator({"a": 1, "b": 0, "operation": "divide"})


def test_unknown_operation_raises_value_error():
    with pytest.raises(ValueError, match="does not support operation"):
        run_calculator({"a": 1, "b": 2, "operation": "power"})


@pytest.mark.parametrize(
    "arguments",
    [
        {"b": 2, "operation": "add"},
        {"a": 1, "operation": "add"},
        {"a": 1, "b": 2},
    ],
)
def test_missing_argument_raises_value_error(arguments):
    with pytest.raises(ValueError, match="missing the"):
        run_calculator(arguments)


@pytest.mark.parametrize("value", ["17", None, True, [17], {"x": 1}])
def test_non_numeric_operand_raises_value_error(value):
    with pytest.raises(ValueError, match="must be a number"):
        run_calculator({"a": value, "b": 2, "operation": "add"})


# ------------------------------------------------------------------ executor


def test_executor_runs_the_calculator_and_stringifies_the_result():
    call = ToolCall(
        id="call-1",
        name="calculator",
        arguments={"a": 17, "b": 23, "operation": "multiply"},
    )

    result = build_executor().execute(call)

    assert result.tool_call_id == "call-1"
    assert result.name == "calculator"
    assert result.output == "391"
    assert result.is_error is False


def test_executor_wraps_a_calculator_failure():
    call = ToolCall(
        id="call-1",
        name="calculator",
        arguments={"a": 1, "b": 0, "operation": "divide"},
    )

    with pytest.raises(ToolExecutionError) as info:
        build_executor().execute(call)

    assert isinstance(info.value.__cause__, ValueError)


# ----------------------------------------------------------------- isolation


def test_module_imports_only_the_tool_contract():
    module = importlib.import_module(MODULE_NAME)

    assert _imports_of_path(pathlib.Path(module.__file__)) == [
        "__future__",
        "typing",
        ".tools",
    ]


@pytest.mark.parametrize(
    "token",
    [
        "openai",
        "deepseek",
        "anthropic",
        "llm",
        "worker",
        "tool_loop",
        "ToolLoop",
        "ToolExecutor",
        "orchestrator",
        "planner",
        "subprocess",
        "importlib",
    ],
)
def test_module_never_imports_a_forbidden_name(token):
    module = importlib.import_module(MODULE_NAME)
    imported = _imports_of_path(pathlib.Path(module.__file__))

    assert not any(token in name for name in imported)


def test_calculator_never_calls_eval_or_exec():
    called: set[str] = set()
    for node in ast.walk(ast.parse(module_source())):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)

    assert "eval" not in called
    assert "exec" not in called
    assert "compile" not in called