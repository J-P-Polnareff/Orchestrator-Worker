"""Worker contract for the Orchestrator-Worker pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..capability import WorkerCapability


class WorkerError(RuntimeError):
    """Raised when a worker cannot produce a result."""


class BaseWorker(ABC):
    """Base class every worker implements.

    Contract:
      * ``name``           - stable identifier the orchestrator routes on.
      * ``capability``     - the worker's declared :class:`WorkerCapability`.
      * ``execute(task)``  - run one task and return the result text.

    A worker receives its dependencies (for example an ``LLMClient``) through
    its own constructor, so nothing in this contract depends on a specific
    model provider. The capability is derived from ``name`` and
    ``capability_description``, so a subclass only overrides that class
    attribute; the registry reads the capability straight off the worker.
    """

    name: str = "base"
    capability_description: str = (
        "A general-purpose worker that handles a broad class of tasks."
    )

    @property
    def capability(self) -> WorkerCapability:
        """Return this worker's declared capability."""
        return WorkerCapability(self.name, self.capability_description)

    @abstractmethod
    def execute(self, task: str) -> str:
        """Run ``task`` and return the worker output."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"