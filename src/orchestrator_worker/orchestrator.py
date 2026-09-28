"""Orchestrator: route a task to a worker, plan, and execute a plan.

Flow (Phase 2, unchanged)::

    User Task -> Orchestrator -> WorkerRegistry -> ResearchWorker / CodingWorker
              -> Worker Result -> Orchestrator -> Final Result

Flow (Phase 3, planning only)::

    User Task -> Orchestrator -> Planner -> Plan        (stops here)

Flow (Phase 4A/4B, plan execution)::

    Plan -> Orchestrator -> Router -> Worker (per step)
         -> AgentState -> ExecutionResult

Flow (Phase 4C, execution batch)::

    Plan -> ExecutionResult[] -> ExecutionContext

Each plan step is one independent worker call and steps run strictly in order.
Every result carries its own ``PlanStep`` metadata, so a result never has to be
matched back to its step by position. There is no retry, no evaluator, no
aggregator and no parallel execution yet.
"""

from __future__ import annotations

from .context import ExecutionContext
from .execution import ExecutionResult
from .plan import Plan
from .planner.base import Planner
from .router import Router
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
        self._router = Router(registry)
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

    def execute_plan(self, plan: Plan) -> list[ExecutionResult]:
        """Execute ``plan`` step by step, in order, and return one result per step.

        Every step is routed independently through the router, which looks the
        worker up by ``step.worker_name``; the planner is not consulted again.
        Steps run strictly sequentially and each result keeps the step it came
        from, so results are returned in the same order as ``plan.steps``.

        Raises:
            RegistryError: a step names a worker that is not registered.
            WorkerError: a worker failed while handling its step. Both errors
                are propagated unchanged, with no fallback and no retry.
        """
        results: list[ExecutionResult] = []
        for step in plan.steps:
            worker = self._router.resolve(step.worker_name)
            output = worker.execute(step.task)

            state = AgentState(user_task=step.task)
            state.worker_name = worker.name
            state.worker_input = step.task
            state.worker_output = output
            state.final_result = output

            results.append(
                ExecutionResult(
                    step_id=step.id,
                    task=step.task,
                    worker_name=worker.name,
                    state=state,
                )
            )
        return results

    def execute_plan_context(self, plan: Plan) -> ExecutionContext:
        """Execute ``plan`` and return its results as an :class:`ExecutionContext`.

        This is a thin wrapper over :meth:`execute_plan`: it runs no worker
        itself, never consults the planner and leaves ``plan`` untouched. The
        context validates that the results describe exactly ``plan``.

        Raises:
            RegistryError: a step names a worker that is not registered.
            WorkerError: a worker failed while handling its step.
            ExecutionContextError: the results do not describe ``plan``.
        """
        return ExecutionContext(plan=plan, results=tuple(self.execute_plan(plan)))

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