"""ResearchToolWorker: the concrete tool-enabled research worker.

A thin subclass of :class:`~orchestrator_worker.workers.tool_enabled.ToolEnabledWorker`.
It supplies the research identity and prompt; the injected ``ToolLoop`` owns
every model call and every tool call, so this worker never builds a loop, a
registry or a model client of its own and never knows a provider.

It keeps the worker name ``"research"`` so the planner contract does not change:
the existing tool-free :class:`~orchestrator_worker.workers.research.ResearchWorker`
stays available, but the application composition root registers this worker for
``"research"`` instead.
"""

from __future__ import annotations

from .tool_enabled import ToolEnabledWorker

DEFAULT_SYSTEM_PROMPT = (
    "You are a research worker. Answer the task with a clear, well-structured "
    "response. You may use the provided tools when they help: for arithmetic, "
    "call the calculator instead of working it out yourself, never guess or "
    "invent a tool result, and wait for the tool result before giving your "
    "final answer. State your reasoning briefly, and if you are unsure about a "
    "fact, say so explicitly instead of inventing details. Return only the "
    "answer the task asks for."
)


class ResearchToolWorker(ToolEnabledWorker):
    """A research worker whose task runs through an injected ``ToolLoop``."""

    name = "research"
    system_prompt = DEFAULT_SYSTEM_PROMPT