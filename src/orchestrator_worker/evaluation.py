"""Evaluation results: whether executed steps can support a final answer.

This module is pure data. It never talks to an LLM, never executes anything and
never retries anything; an evaluator builds an :class:`EvaluationResult` after
it has judged an :class:`~orchestrator_worker.context.ExecutionContext`.

An :class:`EvaluationResult` carries the overall verdict plus an optional
per-step breakdown (:class:`StepEvaluation`). The failed step ids are derived
from that breakdown, never stored separately.
"""

from __future__ import annotations

from dataclasses import dataclass


class EvaluationError(RuntimeError):
    """Raised when an evaluation result is invalid, or an evaluation failed."""


@dataclass(frozen=True)
class StepEvaluation:
    """The verdict for one plan step.

    ``feedback`` may be empty for a step that passed; it only has to explain a
    failure. Like the plan it refers to, a step evaluation is immutable and
    carries no retry counter, attempt number, score or confidence.
    """

    step_id: str
    passed: bool
    feedback: str

    def __post_init__(self) -> None:
        if not isinstance(self.step_id, str):
            raise EvaluationError(
                "StepEvaluation.step_id must be a string, "
                f"got {type(self.step_id).__name__}."
            )
        if not self.step_id.strip():
            raise EvaluationError("StepEvaluation.step_id must be non-empty.")
        if not isinstance(self.passed, bool):
            raise EvaluationError(
                "StepEvaluation.passed must be a bool, "
                f"got {type(self.passed).__name__}."
            )
        if not isinstance(self.feedback, str):
            raise EvaluationError(
                "StepEvaluation.feedback must be a string, "
                f"got {type(self.feedback).__name__}."
            )


@dataclass(frozen=True)
class EvaluationResult:
    """The verdict of one evaluation run.

    ``passed``/``reason`` are the overall verdict; ``step_evaluations`` is the
    optional per-step breakdown. A result without step evaluations is a valid
    whole-result verdict, so existing constructions keep working. When step
    evaluations are present an overall pass may not coexist with a failed step:
    that combination is a structural contradiction, not something to repair.
    """

    passed: bool
    reason: str
    step_evaluations: tuple[StepEvaluation, ...] = ()

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
        if not isinstance(self.step_evaluations, tuple):
            raise EvaluationError(
                "EvaluationResult.step_evaluations must be a tuple, "
                f"got {type(self.step_evaluations).__name__}."
            )

        seen: set[str] = set()
        for index, step_evaluation in enumerate(self.step_evaluations):
            if not isinstance(step_evaluation, StepEvaluation):
                raise EvaluationError(
                    f"EvaluationResult.step_evaluations[{index}] must be a "
                    f"StepEvaluation, got {type(step_evaluation).__name__}."
                )
            if step_evaluation.step_id in seen:
                raise EvaluationError(
                    "EvaluationResult.step_evaluations contain duplicate step "
                    f"id {step_evaluation.step_id!r}."
                )
            seen.add(step_evaluation.step_id)

        if self.passed and any(
            not step_evaluation.passed for step_evaluation in self.step_evaluations
        ):
            raise EvaluationError(
                "EvaluationResult cannot pass overall while a step is marked "
                "as failed."
            )

    @property
    def failed_step_ids(self) -> tuple[str, ...]:
        """Return the ids of the failed steps, in evaluation order.

        Derived straight from ``step_evaluations`` so there is never a second
        piece of failed-step state to keep in sync.
        """
        return tuple(
            step_evaluation.step_id
            for step_evaluation in self.step_evaluations
            if not step_evaluation.passed
        )