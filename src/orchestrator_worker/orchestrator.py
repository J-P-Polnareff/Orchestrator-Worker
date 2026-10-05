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

Flow (Phase 5E, targeted retry)::

    Plan -> execute_plan_context -> Evaluator -> passed? -> return context
         -> failed? -> re-execute the failed steps only -> Evaluator
         -> budget spent or no failed step left? -> RetryError

Flow (Phase 4F, full pipeline)::

    User Task -> plan -> Plan -> execute_plan_with_retry -> ExecutionContext
              -> aggregate -> Final Answer

Each plan step is one independent worker call and steps run strictly in order.
Every result carries its own ``PlanStep`` metadata, so a result never has to be
matched back to its step by position. Every step - whether it runs as part of a
whole plan, on its own, or again during a targeted retry - goes through
:meth:`Orchestrator.execute_step`, so there is exactly one worker-dispatch path.
Retries are evaluator-driven and target only the steps the evaluator marked as
failed; a failure that cannot be tied to any plan step fails explicitly instead
of re-running the plan. There is no replanning and no parallel execution.
"""

from __future__ import annotations

from .aggregators.base import Aggregator
from .context import ExecutionContext
from .evaluators.base import Evaluator
from .execution import ExecutionResult
from .plan import Plan, PlanStep
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

        Every step is executed by :meth:`execute_step`, so the plan loop and
        any single-step re-execution share one worker-dispatch path. The step is
        routed independently through the router, which looks the worker up by
        ``step.worker_name``; the planner is not consulted again. Steps run
        strictly sequentially and each result keeps the step it came from, so
        results are returned in the same order as ``plan.steps``.

        Raises:
            RegistryError: a step names a worker that is not registered.
            WorkerError: a worker failed while handling its step. Both errors
                are propagated unchanged, with no fallback and no retry.
        """
        return [self.execute_step(step) for step in plan.steps]

    def execute_step(self, step: PlanStep) -> ExecutionResult:
        """Execute exactly one plan step and return its result.

        This is the single-step execution primitive that :meth:`execute_plan`
        builds on, so the plan loop and any future single-step re-execution
        share one worker-dispatch implementation. It resolves
        ``step.worker_name`` through the deterministic router, runs that worker
        once and wraps the outcome in an :class:`ExecutionResult`. It never
        touches the planner, evaluator, aggregator or any retry logic, and it
        never executes another step.

        Raises:
            RegistryError: the step names a worker that is not registered.
            WorkerError: the worker failed while handling its step.
        """
        worker = self._router.resolve(step.worker_name)
        output = worker.execute(step.task)

        state = AgentState(user_task=step.task)
        state.worker_name = worker.name
        state.worker_input = step.task
        state.worker_output = output
        state.final_result = output

        return ExecutionResult(
            step_id=step.id,
            task=step.task,
            worker_name=worker.name,
            state=state,
        )

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
        """Execute ``plan``, evaluate it, and re-run only the failed steps.

        The plan is executed once through :meth:`execute_plan_context` and then
        evaluated. On ``passed=False`` the retry controller reads
        :attr:`EvaluationResult.failed_step_ids` and re-executes exactly those
        steps - in plan order, each through :meth:`execute_step` - before
        evaluating the updated context once. Steps that passed are never run
        again, and the plan is never modified and never re-planned.

        ``max_retries`` counts the targeted retry rounds *after* the first
        execution, so ``max_retries=0`` fails right after the initial
        evaluation, ``max_retries=1`` allows one retry round and so on.

        A failure whose ``failed_step_ids`` is empty is not retryable; it is
        reported as a ``RetryError`` rather than re-running the whole plan,
        because there is no step the controller can safely re-execute. If
        ``failed_step_ids`` names a step that is not in the plan, the round is
        rejected before anything runs, so a failure is never partially retried.

        Only ``EvaluationResult.passed is False`` starts another round. Errors
        raised while executing a step or while evaluating it (``WorkerError``,
        ``RegistryError``, ``EvaluatorError``, ``EvaluationError``) propagate
        unchanged and never trigger a retry.

        Raises:
            OrchestratorError: ``plan`` is not a Plan, no evaluator was
                configured, or ``max_retries`` is not a non-negative int.
            RetryError: evaluation failed with no retryable step, or the retry
                budget was exhausted before the results passed.
            RegistryError: a step names a worker that is not registered.
            WorkerError: a worker failed while handling its step.
            EvaluatorError: the evaluator could not produce a verdict.
        """
        if not isinstance(plan, Plan):
            raise OrchestratorError(f"plan must be a Plan, got {type(plan).__name__}.")
        attempts_allowed = self._validated_attempt_budget(max_retries)
        if self._evaluator is None:
            raise OrchestratorError(
                "no evaluator configured; pass evaluator=... when constructing "
                "Orchestrator to use execute_plan_with_retry()."
            )

        context = self.execute_plan_context(plan)

        attempts = 0
        while True:
            evaluation = self._evaluator.evaluate(context)
            attempts += 1
            if evaluation.passed:
                return context

            failed_step_ids = evaluation.failed_step_ids
            if not failed_step_ids:
                raise RetryError(
                    "evaluation failed but no retryable step was identified: "
                    f"{evaluation.reason}"
                )
            planned_step_ids = {step.id for step in plan.steps}
            unknown_step_ids = [
                step_id
                for step_id in failed_step_ids
                if step_id not in planned_step_ids
            ]
            if unknown_step_ids:
                raise RetryError(
                    "evaluation failed and named steps that are not in the "
                    f"plan: {', '.join(unknown_step_ids)}; "
                    "no step was re-executed."
                )
            if attempts >= attempts_allowed:
                raise RetryError(
                    f"evaluation failed after {attempts} attempt(s): "
                    f"{evaluation.reason}"
                )

            context = self._retry_failed_steps(plan, context, failed_step_ids)

    def _retry_failed_steps(
        self,
        plan: Plan,
        context: ExecutionContext,
        failed_step_ids: tuple[str, ...],
    ) -> ExecutionContext:
        """Re-execute the failed steps in plan order and return a new context.

        Every step is re-run through :meth:`execute_step`, so the retry path and
        the initial plan loop share one worker-dispatch implementation. Each
        fresh result replaces that step's current result through
        :meth:`ExecutionContext.replace_step_result`, so a step only ever keeps
        one current result and ``plan`` is never changed.
        """
        failed = set(failed_step_ids)
        updated = context
        for step in plan.steps:
            if step.id in failed:
                result = self.execute_step(step)
                updated = updated.replace_step_result(step.id, result)
        return updated

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
        and re-executes only the failed steps, and only the context that
        passed evaluation is handed to :meth:`aggregate`. Nothing is cached,
        re-planned or run in parallel, and no exception is swallowed.

        ``max_retries`` is validated by :meth:`execute_plan_with_retry`; a
        missing planner, evaluator or aggregator raises the same
        ``OrchestratorError`` those methods already raise.

        Raises:
            OrchestratorError: no planner, evaluator or aggregator configured,
                or ``max_retries`` is not a non-negative int.
            RetryError: evaluation failed with no retryable step, or the
                retry budget is exhausted; the aggregator is not called.
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