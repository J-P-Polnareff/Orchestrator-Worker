"""Minimal Phase 1 orchestrator.

Flow::

    User Task -> Orchestrator -> ResearchWorker -> Worker Result
              -> Orchestrator -> Final Result

Routing is fixed to the single worker injected at construction time. There is
no planner, no router, no evaluator and no aggregator yet.
"""

from __future__ import annotations

from .state import AgentState
from .workers.base import BaseWorker, WorkerError


class OrchestratorError(RuntimeError):
    """Raised when the orchestration flow cannot complete."""


class Orchestrator:
    """Runs one user task through one worker and returns the finished state."""

    def __init__(self, worker: BaseWorker) -> None:
        self._worker = worker

    @property
    def worker_name(self) -> str:
        """Name of the worker this orchestrator routes to."""
        return self._worker.name

    def run(self, user_task: str) -> AgentState:
        """Route ``user_task`` to the worker and return the completed state."""
        task = (user_task or "").strip()
        if not task:
            raise OrchestratorError("user_task must be a non-empty string.")

        state = AgentState(user_task=task)
        state.worker_name = self._worker.name
        state.worker_input = task

        try:
            state.worker_output = self._worker.execute(state.worker_input)
        except WorkerError as exc:
            raise OrchestratorError(
                f"Worker {self._worker.name!r} failed while handling the task: {exc}"
            ) from exc

        state.final_result = state.worker_output
        return state