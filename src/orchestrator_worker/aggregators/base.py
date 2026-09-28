"""Aggregator abstraction: turn a finished execution into a final answer.

An aggregator only ever combines results that already exist. It must not plan,
route, execute or retry anything, and it must not depend on a concrete LLM
provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..context import ExecutionContext


class AggregatorError(RuntimeError):
    """Raised when a final answer cannot be produced."""


class Aggregator(ABC):
    """Turns an :class:`ExecutionContext` into one final answer."""

    @abstractmethod
    def aggregate(self, context: ExecutionContext) -> str:
        """Return the final answer for ``context``. Must not execute anything."""