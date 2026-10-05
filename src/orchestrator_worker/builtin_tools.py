"""Provider-neutral built-in tools.

Concrete tool implementations only: each factory returns a plain
:class:`~orchestrator_worker.tools.Tool` whose handler is a self-contained,
deterministic Python function. Nothing here talks to a model, a worker or the
tool runtime, so the same definition can be offered through any provider.

The calculator is deliberately closed over a fixed set of operations: it never
calls ``eval``, never executes arbitrary code and never guesses.
"""

from __future__ import annotations

from typing import Any

from .tools import Tool

CALCULATOR_NAME = "calculator"

CALCULATOR_OPERATIONS = ("add", "subtract", "multiply", "divide")

CALCULATOR_DESCRIPTION = (
    "Perform one basic arithmetic operation on two numbers. Pass the operation "
    "as 'add', 'subtract', 'multiply' or 'divide' together with the operands "
    "'a' and 'b'. Use it instead of working the arithmetic out yourself."
)

CALCULATOR_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "a": {"type": "number"},
        "b": {"type": "number"},
        "operation": {
            "type": "string",
            "enum": list(CALCULATOR_OPERATIONS),
        },
    },
    "required": ["a", "b", "operation"],
    "additionalProperties": False,
}


def _require_number(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"calculator operand {field!r} must be a number, "
            f"got {type(value).__name__}"
        )


def _calculate(arguments: dict[str, Any]) -> int | float:
    """Run one arithmetic operation described by ``arguments``.

    Raises:
        ValueError: an operand is missing or not a number, the operation is
            unknown, or the divisor is zero. The tool runtime turns a failing
            handler into an error tool result, so the model can react.
    """
    try:
        operation = arguments["operation"]
        a = arguments["a"]
        b = arguments["b"]
    except KeyError as exc:
        raise ValueError(f"calculator is missing the {exc.args[0]!r} argument") from exc

    _require_number(a, "a")
    _require_number(b, "b")

    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            raise ValueError("calculator cannot divide by zero")
        return a / b
    raise ValueError(f"calculator does not support operation {operation!r}")


def calculator_tool() -> Tool:
    """Return the built-in calculator :class:`Tool`."""
    return Tool(
        name=CALCULATOR_NAME,
        description=CALCULATOR_DESCRIPTION,
        parameters=CALCULATOR_PARAMETERS,
        handler=_calculate,
    )