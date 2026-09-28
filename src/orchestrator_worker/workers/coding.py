"""CodingWorker: answers a task with coding-oriented guidance.

Like ResearchWorker this is a thin LLM worker. It does not execute code, does
not run shell commands and does not call external APIs; it only shapes the
prompt. It depends on the LLMClient abstraction, never on a concrete provider.
"""

from __future__ import annotations

from ..llm import LLMClient, LLMError, LLMRequest, LLMResponse, Message
from .base import BaseWorker, WorkerError

DEFAULT_SYSTEM_PROMPT = (
    "You are a coding worker. Answer the task with practical software "
    "engineering guidance: describe the approach, then show a small, complete "
    "example where it helps. You cannot run code or access a shell, so never "
    "claim to have executed anything. If you are unsure about a detail, say so "
    "explicitly instead of inventing it."
)


class CodingWorker(BaseWorker):
    """Produces coding-oriented guidance for a task."""

    name = "coding"

    def __init__(
        self,
        llm: LLMClient,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._llm = llm
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens

    def execute(self, task: str) -> str:
        task = (task or "").strip()
        if not task:
            raise WorkerError(f"{self.name} worker received an empty task.")

        request = LLMRequest(
            messages=[Message.system(self._system_prompt), Message.user(task)],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        try:
            response: LLMResponse = self._llm.complete(request)
        except LLMError as exc:
            raise WorkerError(f"{self.name} worker LLM call failed: {exc}") from exc

        content = (response.content or "").strip()
        if not content:
            raise WorkerError(
                f"{self.name} worker received an empty response from the LLM."
            )
        return content