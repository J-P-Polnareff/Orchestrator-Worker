"""ResearchWorker: answers a task using the provider-agnostic LLM client.

Phase 1 scope is a single LLM completion. There is no web search and no tool
calling, and no dependency on a concrete provider such as DeepSeekClient.
"""

from __future__ import annotations

from ..llm import LLMClient, LLMError, LLMRequest, LLMResponse, Message
from .base import BaseWorker, WorkerError

DEFAULT_SYSTEM_PROMPT = (
    "You are a research worker. Answer the task with a clear, well-structured "
    "response. State your reasoning briefly, and if you are unsure about a fact, "
    "say so explicitly instead of inventing details."
)


class ResearchWorker(BaseWorker):
    """Produces a written answer for a research-style task."""

    name = "research"

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