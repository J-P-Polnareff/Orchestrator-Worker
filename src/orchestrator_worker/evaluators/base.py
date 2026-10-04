"""Evaluator abstraction: judge whether executed results are good enough.

An evaluator only ever inspects an :class:`ExecutionContext`. It must not plan,
execute, retry or aggregate, and it must not depend on a concrete LLM provider,
on a worker, on the router or on the aggregator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..context import ExecutionContext
from ..evaluation import EvaluationError, EvaluationResult


class EvaluatorError(EvaluationError):
    """Raised when an evaluator cannot complete an evaluation."""


class Evaluator(ABC):
    """Judges an :class:`ExecutionContext` and returns an EvaluationResult."""

    @abstractmethod
    def evaluate(self, context: ExecutionContext) -> EvaluationResult:
        """Judge ``context``. Must not execute, retry or modify anything."""