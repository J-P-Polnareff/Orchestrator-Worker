"""Provider-neutral tool foundation: definitions, calls, results, registry.

This module is pure infrastructure. It never talks to an LLM, never imports a
provider adapter and never touches workers, planners, evaluators, aggregators or
the orchestrator. A tool is described by plain data and executed by a plain
callable, so the same definitions can later be offered to any provider.

Nothing here performs tool calling: this layer only defines the contract
(``Tool`` -> ``ToolCall`` -> ``ToolExecutor`` -> ``ToolResult``) that a later
stage can drive from a model response.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


class ToolError(RuntimeError):
    """Base class for tool definition, registry and execution failures."""


class ToolRegistryError(ToolError):
    """Raised when a tool cannot be registered or looked up."""


class ToolExecutionError(ToolError):
    """Raised when a tool handler fails while executing a call."""


def _validate_text(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise ToolError(f"{field} must be a string, got {type(value).__name__}.")
    if not value.strip():
        raise ToolError(f"{field} must be a non-empty string.")


@dataclass(frozen=True)
class Tool:
    """A provider-neutral tool definition.

    ``parameters`` carries the input schema as a plain mapping and is stored
    as-is; nothing here validates it. ``handler`` receives the already-parsed
    argument mapping and may return any object.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: Callable[[dict[str, Any]], Any]

    def __post_init__(self) -> None:
        _validate_text(self.name, "Tool.name")
        _validate_text(self.description, "Tool.description")
        if not isinstance(self.parameters, Mapping):
            raise ToolError(
                "Tool.parameters must be a mapping, "
                f"got {type(self.parameters).__name__}."
            )
        if not callable(self.handler):
            raise ToolError(
                f"Tool.handler must be callable, got {type(self.handler).__name__}."
            )


@dataclass(frozen=True)
class ToolCall:
    """A parsed request to run a tool.

    ``arguments`` is already a plain ``dict``: provider SDK objects are never
    kept here.
    """

    id: str
    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        _validate_text(self.id, "ToolCall.id")
        _validate_text(self.name, "ToolCall.name")
        if not isinstance(self.arguments, dict):
            raise ToolError(
                "ToolCall.arguments must be a dict, "
                f"got {type(self.arguments).__name__}."
            )


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one tool call, always as plain text."""

    tool_call_id: str
    name: str
    output: str
    is_error: bool = False

    def __post_init__(self) -> None:
        _validate_text(self.tool_call_id, "ToolResult.tool_call_id")
        _validate_text(self.name, "ToolResult.name")
        if not isinstance(self.output, str):
            raise ToolError(
                "ToolResult.output must be a string, "
                f"got {type(self.output).__name__}."
            )
        if not isinstance(self.is_error, bool):
            raise ToolError(
                "ToolResult.is_error must be a bool, "
                f"got {type(self.is_error).__name__}."
            )


class ToolRegistry:
    """Maps tool names to :class:`Tool` definitions.

    Duplicate names are rejected rather than silently overwritten, and an
    unknown lookup never falls back to another tool. The registry only knows
    the :class:`Tool` data type, never a concrete tool.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add ``tool`` under its ``name``.

        Raises:
            ToolRegistryError: ``tool`` is not a Tool, or its name is taken.
        """
        if not isinstance(tool, Tool):
            raise ToolRegistryError(
                f"register() expects a Tool, got {type(tool).__name__}."
            )
        if tool.name in self._tools:
            raise ToolRegistryError(
                f"tool {tool.name!r} is already registered; "
                "duplicate names are not allowed."
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return the tool registered under ``name``.

        Raises:
            ToolRegistryError: if no tool with that name is registered. There
                is intentionally no fallback to another tool.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise ToolRegistryError(
                f"unknown tool {name!r}; registered tools: {self._describe()}"
            ) from None

    def has(self, name: str) -> bool:
        """Return True if a tool is registered under ``name``."""
        return name in self._tools

    def names(self) -> list[str]:
        """Return the registered tool names, sorted."""
        return sorted(self._tools)

    def tools(self) -> tuple[Tool, ...]:
        """Return the registered tools, sorted by name, as a read-only tuple."""
        return tuple(self._tools[name] for name in sorted(self._tools))

    def _describe(self) -> str:
        return ", ".join(self.names()) if self._tools else "(none)"


class ToolExecutor:
    """Runs a :class:`ToolCall` against a :class:`ToolRegistry`.

    Executing never guesses: an unknown tool raises ``ToolRegistryError`` and a
    failing handler raises ``ToolExecutionError`` with the original exception as
    its ``__cause__``. Handler output is normalized with ``str()``.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        if not isinstance(registry, ToolRegistry):
            raise ToolError(
                "ToolExecutor expects a ToolRegistry, "
                f"got {type(registry).__name__}."
            )
        self._registry = registry

    def execute(self, call: ToolCall) -> ToolResult:
        """Run ``call`` and return its :class:`ToolResult`.

        Raises:
            ToolError: ``call`` is not a ToolCall.
            ToolRegistryError: ``call.name`` is not registered.
            ToolExecutionError: the handler raised while executing.
        """
        if not isinstance(call, ToolCall):
            raise ToolError(
                f"execute() expects a ToolCall, got {type(call).__name__}."
            )

        tool = self._registry.get(call.name)

        try:
            raw = tool.handler(call.arguments)
        except Exception as exc:
            raise ToolExecutionError(
                f"tool {call.name!r} failed while executing call {call.id!r}: {exc}"
            ) from exc

        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            output=str(raw),
            is_error=False,
        )