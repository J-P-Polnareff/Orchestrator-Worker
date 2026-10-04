"""Evaluation results: whether executed steps can support a final answer.

This module is pure data. It never talks to an LLM, never executes anything and
never retries anything; an evaluator builds an :class:`EvaluationResult` after
it has judged an :class:`~orchestrator_worker.context.ExecutionContext`.
"""

from __future__ import annotations

from dataclasses import dataclass


class EvaluationError(RuntimeError):
    """Raised when an evaluation result is invalid, or an evaluation failed."""


@dataclass(frozen=True)
class EvaluationResult:
    """The verdict of one evaluation run.

    It deliberately carries no score, status or retry counter: this phase only
    answers "are the executed results good enough for a final answer, and why".
    """

    passed: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise EvaluationError(
                "EvaluationResult.passed must be a bool, "
                f"got {type(self.passed).__name__}."
            )
        if not isinstance(self.reason, str):
            raise EvaluationError(
                "EvaluationResult.reason must be a string, "
                f"got {type(self.reason).__name__}."
            )
        if not self.reason.strip():
            raise EvaluationError(
                "EvaluationResult.reason must be a non-empty string."
            )