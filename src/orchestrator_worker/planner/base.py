"""Planner abstraction: turn a natural-language task into a Plan.

A planner only ever produces a plan. It must not execute workers, and it must
not depend on a concrete worker or a concrete LLM provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..plan import Plan


class PlannerError(RuntimeError):
    """Base class for every planning failure."""


class PlannerLLMError(PlannerError):
    """The underlying LLM call failed."""


class PlanParsingError(PlannerError):
    """The LLM response could not be read as a JSON object."""


class PlanValidationError(PlannerError):
    """The parsed JSON is not a valid plan."""


class UnknownWorkerError(PlanValidationError):
    """A step references a worker that is not available."""


class Planner(ABC):
    """Produces a :class:`~orchestrator_worker.plan.Plan` from a task."""

    @abstractmethod
    def create_plan(self, task: str) -> Plan:
        """Return a plan for ``task``. Must not execute anything."""