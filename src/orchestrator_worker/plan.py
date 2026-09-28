"""Structured plan produced by a planner.

This module is pure data: it never talks to an LLM and never runs a worker.
"""

from __future__ import annotations

from dataclasses import dataclass


class PlanError(RuntimeError):
    """Raised when a Plan or PlanStep is structurally invalid."""


def _validate_text(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise PlanError(f"{field} must be a string, got {type(value).__name__}.")
    if not value.strip():
        raise PlanError(f"{field} must be a non-empty string.")


@dataclass(frozen=True)
class PlanStep:
    """One unit of work inside a plan."""

    id: str
    task: str
    worker_name: str

    def __post_init__(self) -> None:
        _validate_text(self.id, "PlanStep.id")
        _validate_text(self.task, "PlanStep.task")
        _validate_text(self.worker_name, "PlanStep.worker_name")


@dataclass(frozen=True)
class Plan:
    """An ordered list of steps that together satisfy ``goal``."""

    goal: str
    steps: list[PlanStep]

    def __post_init__(self) -> None:
        _validate_text(self.goal, "Plan.goal")
        if not isinstance(self.steps, list):
            raise PlanError(
                f"Plan.steps must be a list, got {type(self.steps).__name__}."
            )
        if not self.steps:
            raise PlanError("Plan.steps must not be empty.")
        for index, step in enumerate(self.steps):
            if not isinstance(step, PlanStep):
                raise PlanError(
                    f"Plan.steps[{index}] must be a PlanStep, "
                    f"got {type(step).__name__}."
                )