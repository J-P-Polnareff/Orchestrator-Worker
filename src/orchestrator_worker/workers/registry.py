"""In-memory registry of workers, keyed by worker name."""

from __future__ import annotations

from .base import BaseWorker, WorkerCapability


class RegistryError(RuntimeError):
    """Raised when a worker cannot be registered or looked up."""


class WorkerRegistry:
    """Maps worker names to worker instances.

    The registry only knows about the :class:`BaseWorker` contract, never about
    concrete workers, so new workers can be added without changing it.

    It is also the single source of truth for worker capabilities: every
    capability is read off the registered worker, so there is no second
    capability list that could drift out of sync with the registrations.

    Duplicate names are rejected rather than silently overwritten: an existing
    registration must be removed explicitly by the caller if that is intended.
    """

    def __init__(self) -> None:
        self._workers: dict[str, BaseWorker] = {}

    def register(self, worker: BaseWorker) -> None:
        """Add ``worker`` under its ``name``.

        Raises:
            RegistryError: if a worker with the same name is already registered.
        """
        name = worker.name
        if name in self._workers:
            raise RegistryError(
                f"worker {name!r} is already registered; "
                "duplicate names are not allowed."
            )
        self._workers[name] = worker

    def get(self, name: str) -> BaseWorker:
        """Return the worker registered under ``name``.

        Raises:
            RegistryError: if no worker with that name is registered. There is
                intentionally no fallback to another worker.
        """
        try:
            return self._workers[name]
        except KeyError:
            raise RegistryError(
                f"unknown worker {name!r}; registered workers: {self._describe()}"
            ) from None

    def has(self, name: str) -> bool:
        """Return True if a worker is registered under ``name``."""
        return name in self._workers

    def names(self) -> list[str]:
        """Return the registered worker names, sorted."""
        return sorted(self._workers)

    def capabilities(self) -> list[WorkerCapability]:
        """Return the capability of every registered worker, sorted by name.

        The capability is read from each worker instance, so registering a
        worker is the only step needed to make its capability visible; this is
        what gives the planner dynamic capability awareness.
        """
        return [self._workers[name].capability for name in self.names()]

    def _describe(self) -> str:
        return ", ".join(self.names()) if self._workers else "(none)"