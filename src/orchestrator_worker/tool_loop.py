"""Provider-neutral tool execution loop.

Drives the contract built by the tool foundation and the LLM tool-call
contract::

    LLMRequest -> LLMClient.complete() -> LLMResponse
                                            |
                          response.tool_calls? -- no --> return LLMResponse
                                            |
                                           yes
                                            |
              assistant tool-call Message + one tool Message per result
                                            |
                      new LLMRequest (same tools, longer history)
                                            |
                                   repeat, bounded by max_rounds

The loop is deliberately standalone: it is not wired into a worker, the
orchestrator or the application, it never replans, it never retries an LLM call
and it runs tools strictly sequentially. Its only dependencies are the
provider-neutral ``llm.base`` contract and a plain ``ToolExecutor``; no provider
SDK type is referenced here.
"""

from __future__ import annotations

import dataclasses

from .llm.base import LLMClient, LLMRequest, LLMResponse, Message
from .tools import ToolCall, ToolExecutionError, ToolExecutor, ToolResult


class ToolLoopError(RuntimeError):
    """Raised when the tool loop is misconfigured or out of rounds."""


class ToolLoop:
    """Runs a model/tool exchange until the model stops requesting tools.

    One "round" is one call to ``LLMClient.complete()``. Tool calls are only
    executed when their results can still be delivered back to the model, so a
    tool call that arrives in the final permitted round raises
    :class:`ToolLoopError` instead of running with nowhere to report.
    """

    def __init__(
        self,
        llm: LLMClient,
        executor: ToolExecutor,
        *,
        max_rounds: int = 8,
    ) -> None:
        if not isinstance(llm, LLMClient):
            raise ToolLoopError(
                f"ToolLoop expects an LLMClient, got {type(llm).__name__}."
            )
        if not isinstance(executor, ToolExecutor):
            raise ToolLoopError(
                f"ToolLoop expects a ToolExecutor, got {type(executor).__name__}."
            )
        self._llm = llm
        self._executor = executor
        self._max_rounds = _validate_max_rounds(max_rounds)

    def run(self, request: LLMRequest) -> LLMResponse:
        """Complete ``request``, answering tool calls until the model is done.

        Raises:
            ToolLoopError: ``request`` is not an LLMRequest, the model asked
                for a tool while the request declared none, or ``max_rounds``
                was used up while the model still requested tools.
            LLMError: propagated unchanged from ``LLMClient.complete``.
            ToolRegistryError: propagated unchanged when the model asks for an
                unregistered tool.
        """
        if not isinstance(request, LLMRequest):
            raise ToolLoopError(
                "ToolLoop.run() expects an LLMRequest, "
                f"got {type(request).__name__}."
            )

        current = request
        rounds_left = self._max_rounds
        while True:
            response = self._llm.complete(current)

            if not response.tool_calls:
                return response

            if not current.tools:
                raise ToolLoopError(
                    "the model requested tools but the request declared none; "
                    "tool definitions must be supplied explicitly."
                )

            rounds_left -= 1
            if rounds_left <= 0:
                raise ToolLoopError(
                    f"tool loop exceeded max_rounds={self._max_rounds} while the "
                    "model still requested tools."
                )

            messages = list(current.messages)
            messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            for call in response.tool_calls:
                result = self._execute(call)
                messages.append(Message.tool(result.output, result.tool_call_id))
            current = dataclasses.replace(current, messages=tuple(messages))

    def _execute(self, call: ToolCall) -> ToolResult:
        """Run one call, turning a failing tool into an error result.

        A ``ToolExecutionError`` means the tool itself failed, which the model
        may be able to react to, so it becomes an ``is_error=True`` result
        instead of aborting the loop. Unknown tools are a configuration
        problem and keep propagating as ``ToolRegistryError``.
        """
        try:
            return self._executor.execute(call)
        except ToolExecutionError as exc:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                output=str(exc),
                is_error=True,
            )


def _validate_max_rounds(max_rounds: int) -> int:
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int):
        raise ToolLoopError(
            "ToolLoop.max_rounds must be an int, "
            f"got {type(max_rounds).__name__}."
        )
    if max_rounds < 1:
        raise ToolLoopError(
            f"ToolLoop.max_rounds must be at least 1, got {max_rounds}."
        )
    return max_rounds