"""Deterministic router: resolve a plan step's worker name to a worker.

Flow::

    PlanStep.worker_name -> Router.resolve -> WorkerRegistry.get -> BaseWorker

The router is plain program logic. It never calls an LLM, never rewrites the
plan and never picks a worker other than the one it is asked for. An unknown
name propagates the registry's ``RegistryError`` unchanged.
"""

from __future__ import annotations

from .workers.base import BaseWorker
from .workers.registry import WorkerRegistry


class Router:
    """Turns a worker name into the registered worker instance."""

    def __init__(self, registry: WorkerRegistry) -> None:
        self._registry = registry

    def resolve(self, worker_name: str) -> BaseWorker:
        """Return the worker registered under ``worker_name``.

        Raises:
            RegistryError: no worker is registered under that name. There is
                deliberately no fallback to another worker.
        """
        return self._registry.get(worker_name)