"""Execution results: a plan step paired with what its worker produced.

This module is pure data. It never talks to an LLM, never resolves a worker and
never runs anything; the orchestrator builds an :class:`ExecutionResult` after
a step has already executed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .state import AgentState


class ExecutionError(RuntimeError):
    """Raised when an ExecutionResult is structurally invalid."""


def _validate_text(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise ExecutionError(f"{field} must be a string, got {type(value).__name__}.")
    if not value.strip():
        raise ExecutionError(f"{field} must be a non-empty string.")


@dataclass(frozen=True)
class ExecutionResult:
    """The outcome of one plan step.

    It keeps the step it came from (``step_id``, ``task``, ``worker_name``)
    together with the :class:`~orchestrator_worker.state.AgentState` that the
    worker produced, so a result never has to be matched back to its step by
    list position.
    """

    step_id: str
    task: str
    worker_name: str
    state: AgentState

    def __post_init__(self) -> None:
        _validate_text(self.step_id, "ExecutionResult.step_id")
        _validate_text(self.task, "ExecutionResult.task")
        _validate_text(self.worker_name, "ExecutionResult.worker_name")
        if not isinstance(self.state, AgentState):
            raise ExecutionError(
                "ExecutionResult.state must be an AgentState, "
                f"got {type(self.state).__name__}."
            )