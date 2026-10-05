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

    def replace_step_result(
        self, step_id: str, result: ExecutionResult
    ) -> "ExecutionContext":
        """Return a copy of this context with ``step_id``'s current result replaced.

        A step always has exactly one *current* result: the existing entry is
        swapped for ``result`` in place, never appended, so a step can never
        end up with two results. ``plan`` and every other step's result are
        reused unchanged.

        Raises:
            ExecutionContextError: ``step_id`` is not a non-empty string, the
                replacement is not an :class:`ExecutionResult`, the result
                belongs to a different step, or ``step_id`` is not a step of
                the plan. There is deliberately no implicit append of an
                unknown step.
        """
        if not isinstance(step_id, str) or not step_id.strip():
            raise ExecutionContextError("step_id must be a non-empty string.")
        if not isinstance(result, ExecutionResult):
            raise ExecutionContextError(
                "replacement must be an ExecutionResult, "
                f"got {type(result).__name__}."
            )
        if result.step_id != step_id:
            raise ExecutionContextError(
                f"result for step {result.step_id!r} cannot replace step "
                f"{step_id!r}."
            )

        planned = {step.id for step in self.plan.steps}
        if step_id not in planned:
            raise ExecutionContextError(
                f"cannot replace the result of unknown step {step_id!r}; "
                f"plan steps: {', '.join(sorted(planned)) or '(none)'}."
            )
        if not any(existing.step_id == step_id for existing in self.results):
            raise ExecutionContextError(
                f"step {step_id!r} has no current result to replace."
            )

        replaced = tuple(
            result if existing.step_id == step_id else existing
            for existing in self.results
        )
        return ExecutionContext(plan=self.plan, results=replaced)

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