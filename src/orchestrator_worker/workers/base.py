"""Worker contract for the Orchestrator-Worker pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod


class WorkerError(RuntimeError):
    """Raised when a worker cannot produce a result."""


class BaseWorker(ABC):
    """Base class every worker implements.

    Contract:
      * ``name``           - stable identifier the orchestrator routes on.
      * ``execute(task)``  - run one task and return the result text.

    A worker receives its dependencies (for example an ``LLMClient``) through
    its own constructor, so nothing in this contract depends on a specific
    model provider.
    """

    name: str = "base"

    @abstractmethod
    def execute(self, task: str) -> str:
        """Run ``task`` and return the worker output."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"