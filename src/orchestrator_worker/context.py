"""Execution context: the complete result set produced by one plan.

This module is pure data. It never talks to an LLM, never resolves a worker and
never runs anything; the orchestrator builds an :class:`ExecutionContext` once
a plan has finished executing.

A context is only valid when its results describe exactly the plan it belongs
to: every step is covered, no step is repeated and no result refers to a step
that is not in the plan.
"""

from __future__ import annotations

from dataclasses import dataclass

from .execution import ExecutionResult
from .plan import Plan


class ExecutionContextError(RuntimeError):
    """Raised when an ExecutionContext is structurally invalid."""


@dataclass(frozen=True)
class ExecutionContext:
    """A plan together with the results of executing all of its steps.

    ``results`` is a tuple so the collection cannot be mutated after the
    context is built. The order of the results is not constrained: they may
    come back in any order, they are never re-sorted here.
    """

    plan: Plan
    results: tuple[ExecutionResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, Plan):
            raise ExecutionContextError(
                "ExecutionContext.plan must be a Plan, "
                f"got {type(self.plan).__name__}."
            )
        if not isinstance(self.results, tuple):
            raise ExecutionContextError(
                "ExecutionContext.results must be a tuple, "
                f"got {type(self.results).__name__}."
            )
        for index, result in enumerate(self.results):
            if not isinstance(result, ExecutionResult):
                raise ExecutionContextError(
                    f"ExecutionContext.results[{index}] must be an ExecutionResult, "
                    f"got {type(result).__name__}."
                )
        self._validate_step_coverage()

    def _validate_step_coverage(self) -> None:
        """Check that the results describe exactly the steps of the plan."""
        planned = [step.id for step in self.plan.steps]
        produced = [result.step_id for result in self.results]

        unknown = sorted({step_id for step_id in produced if step_id not in planned})
        if unknown:
            raise ExecutionContextError(
                "ExecutionContext.results reference steps that are not in the plan: "
                f"{', '.join(unknown)}."
            )

        duplicates = sorted(
            {step_id for step_id in produced if produced.count(step_id) > 1}
        )
        if duplicates:
            raise ExecutionContextError(
                "ExecutionContext.results contain duplicate step ids: "
                f"{', '.join(duplicates)}."
            )

        missing = [step_id for step_id in planned if step_id not in produced]
        if missing:
            raise ExecutionContextError(
                "ExecutionContext.results are missing steps from the plan: "
                f"{', '.join(missing)}."
            )