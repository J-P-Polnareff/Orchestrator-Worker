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

Flow (Phase 4D, aggregation)::

    ExecutionContext -> Aggregator -> Final Answer

Flow (Phase 4E, whole-plan retry)::

    Plan -> execute_plan_context -> Evaluator -> passed? -> return context
                                              -> failed? -> execute the plan again

Flow (Phase 4F, full pipeline)::

    User Task -> plan -> Plan -> execute_plan_with_retry -> ExecutionContext
              -> aggregate -> Final Answer

Each plan step is one independent worker call and steps run strictly in order.
Every result carries its own ``PlanStep`` metadata, so a result never has to be
matched back to its step by position. Retries are evaluator-driven and always
re-run the whole plan: there is no replanning, no targeted per-step retry and
no parallel execution.
"""

from __future__ import annotations

from .aggregators.base import Aggregator
from .context import ExecutionContext
from .evaluators.base import Evaluator
from .execution import ExecutionResult
from .plan import Plan
from .planner.base import Planner
from .retry import RetryError
from .router import Router
from .state import AgentState
from .workers.base import BaseWorker, WorkerError
from .workers.registry import WorkerRegistry

DEFAULT_WORKER_NAME = "research"


class OrchestratorError(RuntimeError):
    """Raised when the orchestration flow cannot complete."""


class Orchestrator:
    """Routes a task to a worker, produces plans and aggregates final answers."""

    def __init__(
        self,
        workers: WorkerRegistry | BaseWorker,
        planner: Planner | None = None,
        aggregator: Aggregator | None = None,
        evaluator: Evaluator | None = None,
    ) -> None:
        """Accept a registry, or a single worker for the Phase 1 shorthand.

        Passing a bare :class:`BaseWorker` wraps it in a registry and makes it
        the default target, so ``Orchestrator(worker).run(task)`` keeps working
        exactly as it did before the registry existed.

        ``planner``, ``aggregator`` and ``evaluator`` are optional so earlier
        constructions stay valid; they are only used by :meth:`plan`,
        :meth:`aggregate` and :meth:`execute_plan_with_retry`.
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
        self._aggregator = aggregator
        self._evaluator = evaluator

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

    def aggregate(self, context: ExecutionContext) -> str:
        """Return the final answer for ``context`` via the configured aggregator.

        Nothing is executed here: by the time this is called the context already
        holds every step result, and ``context.plan`` is left untouched.

        Raises:
            OrchestratorError: no aggregator was configured.
            AggregatorError: the aggregator could not produce an answer.
        """
        if self._aggregator is None:
            raise OrchestratorError(
                "no aggregator configured; pass aggregator=... when constructing "
                "Orchestrator to use aggregate()."
            )
        return self._aggregator.aggregate(context)

    def execute_plan_with_retry(
        self,
        plan: Plan,
        *,
        max_retries: int = 1,
    ) -> ExecutionContext:
        """Execute ``plan``, evaluate it, and re-run the whole plan on failure.

        ``max_retries`` counts the extra attempts *after* the first execution,
        so the plan runs at most ``max_retries + 1`` times and is evaluated at
        most ``max_retries + 1`` times. Every attempt re-executes the plan from
        scratch through :meth:`execute_plan_context`: nothing is cached and no
        result is reused. The plan itself is never modified and never re-planned.

        Only ``EvaluationResult.passed is False`` starts another attempt. Errors
        raised while executing the plan or while evaluating it (``WorkerError``,
        ``RegistryError``, ``EvaluatorError``, ``EvaluationError``) propagate
        unchanged and never trigger a retry.

        Raises:
            OrchestratorError: ``plan`` is not a Plan, no evaluator was
                configured, or ``max_retries`` is not a non-negative int.
            RetryError: the evaluator reported ``passed=False`` and no attempts
                are left.
            RegistryError: a step names a worker that is not registered.
            WorkerError: a worker failed while handling its step.
            EvaluatorError: the evaluator could not produce a verdict.
        """
        if not isinstance(plan, Plan):
            raise OrchestratorError(f"plan must be a Plan, got {type(plan).__name__}.")
        attempts = self._validated_attempt_budget(max_retries)
        if self._evaluator is None:
            raise OrchestratorError(
                "no evaluator configured; pass evaluator=... when constructing "
                "Orchestrator to use execute_plan_with_retry()."
            )

        last_reason = ""
        for _ in range(attempts):
            context = self.execute_plan_context(plan)

            evaluation = self._evaluator.evaluate(context)
            if evaluation.passed:
                return context

            last_reason = evaluation.reason

        raise RetryError(
            f"evaluation failed after {attempts} attempt(s): {last_reason}"
        )

    @staticmethod
    def _validated_attempt_budget(max_retries: int) -> int:
        """Return ``max_retries + 1`` attempts, or raise ``OrchestratorError``."""
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise OrchestratorError(
                "max_retries must be an int (bools are not accepted), "
                f"got {type(max_retries).__name__}."
            )
        if max_retries < 0:
            raise OrchestratorError(f"max_retries must be >= 0, got {max_retries}.")
        return max_retries + 1

    def run_pipeline(self, user_task: str, *, max_retries: int = 1) -> str:
        """Run the full pipeline: plan, execute with retry, then aggregate.

        This is the explicit end-to-end flow::

            user_task -> plan -> Plan -> execute_plan_with_retry
                      -> ExecutionContext -> aggregate -> final answer

        It is pure orchestration over the existing high-level methods: the
        planner runs exactly once, :meth:`execute_plan_with_retry` executes
        (and re-executes) the plan and evaluates it, and only the context that
        passed evaluation is handed to :meth:`aggregate`. Nothing is cached,
        re-planned or run in parallel, and no exception is swallowed.

        ``max_retries`` is validated by :meth:`execute_plan_with_retry`; a
        missing planner, evaluator or aggregator raises the same
        ``OrchestratorError`` those methods already raise.

        Raises:
            OrchestratorError: no planner, evaluator or aggregator configured,
                or ``max_retries`` is not a non-negative int.
            RetryError: the evaluator reported ``passed=False`` and the retry
                budget is exhausted; the aggregator is not called.
            PlannerError: the planner could not produce a valid plan.
            RegistryError: a step names a worker that is not registered.
            WorkerError: a worker failed while handling its step.
            EvaluatorError: the evaluator could not produce a verdict.
            AggregatorError: the aggregator could not produce an answer.
        """
        plan = self.plan(user_task)

        context = self.execute_plan_with_retry(plan, max_retries=max_retries)

        return self.aggregate(context)

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