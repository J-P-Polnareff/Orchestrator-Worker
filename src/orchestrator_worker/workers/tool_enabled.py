"""Tool-enabled worker boundary: a worker that runs its task through a ToolLoop.

``ToolEnabledWorker`` is the seam between the worker contract and the tool
runtime. It holds the ``Tool`` definitions it is allowed to request and hands a
fully built ``LLMRequest`` to an injected ``ToolLoop``. The loop owns every
model call and every tool execution, so the worker never reaches past it to run
a tool or to look one up.

Subclasses supply their identity and prompt (``name``, ``system_prompt``, or a
full ``build_request`` override). They inherit the tool wiring and the
``LLMResponse`` -> result conversion. Nothing here knows a concrete provider,
a global tool catalog or the pipeline above the worker.
"""

from __future__ import annotations

from ..llm import LLMError, LLMRequest, LLMResponse, Message
from ..tool_loop import ToolLoop, ToolLoopError
from ..tools import Tool, ToolRegistryError
from .base import BaseWorker, WorkerError

DEFAULT_SYSTEM_PROMPT = (
    "You are a worker that can use tools. Call a tool when it helps answer the "
    "task, then give a final answer once you have enough information."
)


class ToolEnabledWorker(BaseWorker):
    """A worker whose task runs through an injected :class:`ToolLoop`.

    The constructor takes an already built loop plus the exact ``Tool``
    definitions this worker may request. It never builds a loop, a registry or
    a model client of its own, and it never executes a tool directly: the loop
    is the only tool runtime entry point.
    """

    name = "tool_enabled"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def __init__(
        self,
        tool_loop: ToolLoop,
        *,
        tools: tuple[Tool, ...] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        if not isinstance(tool_loop, ToolLoop):
            raise WorkerError(
                "ToolEnabledWorker expects a ToolLoop, "
                f"got {type(tool_loop).__name__}."
            )
        if not isinstance(tools, tuple):
            raise WorkerError(
                "ToolEnabledWorker tools must be a tuple of Tool, "
                f"got {type(tools).__name__}."
            )
        for index, tool in enumerate(tools):
            if not isinstance(tool, Tool):
                raise WorkerError(
                    f"ToolEnabledWorker tools[{index}] must be a Tool, "
                    f"got {type(tool).__name__}."
                )

        self._tool_loop = tool_loop
        self._tools = tools
        self._temperature = temperature
        self._max_tokens = max_tokens

    def build_request(self, task: str) -> LLMRequest:
        """Build the tool-enabled request for ``task``.

        Subclasses shape the prompt by overriding :attr:`system_prompt` or this
        method; the base implementation only appends the worker's allowed tools.
        """
        return LLMRequest(
            messages=[Message.system(self.system_prompt), Message.user(task)],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            tools=self._tools,
        )

    def execute(self, task: str) -> str:
        task = (task or "").strip()
        if not task:
            raise WorkerError(f"{self.name} worker received an empty task.")

        request = self.build_request(task)

        try:
            response: LLMResponse = self._tool_loop.run(request)
        except LLMError as exc:
            raise WorkerError(f"{self.name} worker LLM call failed: {exc}") from exc
        except ToolRegistryError as exc:
            raise WorkerError(
                f"{self.name} worker tool call failed: {exc}"
            ) from exc
        except ToolLoopError as exc:
            raise WorkerError(
                f"{self.name} worker tool loop failed: {exc}"
            ) from exc

        content = (response.content or "").strip()
        if not content:
            raise WorkerError(
                f"{self.name} worker received an empty response from the LLM."
            )
        return content