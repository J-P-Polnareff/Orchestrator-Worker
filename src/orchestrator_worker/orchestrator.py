"""Orchestrator: routes a user task to a worker, and can ask a planner for a plan.

Flow (Phase 2, unchanged)::

    User Task -> Orchestrator -> WorkerRegistry -> ResearchWorker / CodingWorker
              -> Worker Result -> Orchestrator -> Final Result

Flow (Phase 3, planning only)::

    User Task -> Orchestrator -> Planner -> Plan        (stops here)

``plan()`` only produces a :class:`~orchestrator_worker.plan.Plan`. It does not
execute any step of that plan. There is no planner-driven execution, no router,
no evaluator and no aggregator yet.
"""

from __future__ import annotations

from .plan import Plan
from .planner.base import Planner
from .state import AgentState
from .workers.base import BaseWorker, WorkerError
from .workers.registry import WorkerRegistry

DEFAULT_WORKER_NAME = "research"


class OrchestratorError(RuntimeError):
    """Raised when the orchestration flow cannot complete."""


class Orchestrator:
    """Routes one user task to one registered worker, and produces plans."""

    def __init__(
        self,
        workers: WorkerRegistry | BaseWorker,
        planner: Planner | None = None,
    ) -> None:
        """Accept a registry, or a single worker for the Phase 1 shorthand.

        Passing a bare :class:`BaseWorker` wraps it in a registry and makes it
        the default target, so ``Orchestrator(worker).run(task)`` keeps working
        exactly as it did before the registry existed.

        ``planner`` is optional so the Phase 2 construction stays valid; it is
        only used by :meth:`plan`.
        """
        if isinstance(workers, BaseWorker):
            registry = WorkerRegistry()
            registry.register(workers)
            default_worker = workers.name
        else:
            registry = workers
            default_worker = DEFAULT_WORKER_NAME

        self._registry = registry
        self._default_worker = default_worker
        self._planner = planner

    @property
    def worker_name(self) -> str:
        """Name of the worker used when ``run`` is called without one."""
        return self._default_worker

    def plan(self, task: str) -> Plan:
        """Return a plan for ``task`` without executing any worker.

        Raises:
            OrchestratorError: no planner was configured.
            PlannerError: the planner could not produce a valid plan.
        """
        if self._planner is None:
            raise OrchestratorError(
                "no planner configured; pass planner=... when constructing "
                "Orchestrator to use plan()."
            )
        return self._planner.create_plan(task)

    def run(self, user_task: str, worker_name: str | None = None) -> AgentState:
        """Route ``user_task`` to ``worker_name``, defaulting to research.

        Raises:
            OrchestratorError: ``user_task`` is empty.
            RegistryError: the requested worker is not registered. There is
                deliberately no fallback to another worker.
            OrchestratorError: the selected worker failed.
        """
        task = (user_task or "").strip()
        if not task:
            raise OrchestratorError("user_task must be a non-empty string.")

        worker = self._registry.get(worker_name or self._default_worker)

        state = AgentState(user_task=task)
        state.worker_name = worker.name
        state.worker_input = task

        try:
            state.worker_output = worker.execute(state.worker_input)
        except WorkerError as exc:
            raise OrchestratorError(
                f"Worker {worker.name!r} failed while handling the task: {exc}"
            ) from exc

        state.final_result = state.worker_output
        return state