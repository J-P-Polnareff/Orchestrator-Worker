"""LLM-backed aggregator.

Flow::

    ExecutionContext -> prompt -> LLMClient.complete -> final answer text

It depends on the ``LLMClient`` abstraction only. It never plans, routes,
executes or retries anything: by the time it runs, every step has finished.

The prompt is built from ``context.plan.steps`` (the plan's own order) and each
result is looked up by ``step.id``, so a shuffled ``context.results`` cannot
change the order the model sees.
"""

from __future__ import annotations

from ..context import ExecutionContext
from ..execution import ExecutionResult
from ..llm import LLMClient, LLMError, LLMRequest, LLMResponse, Message
from ..plan import PlanStep
from .base import Aggregator, AggregatorError

DEFAULT_SYSTEM_PROMPT = (
    "You are an aggregation component. You receive the results of a plan that "
    "has already been executed, and you write the final answer for the user.\n"
    "\n"
    "Rules:\n"
    "- Use only the step results you are given.\n"
    "- Never plan, re-plan, re-order or re-run any work; the work is finished.\n"
    "- Write the final answer directly, without describing your process.\n"
)


class LLMAggregator(Aggregator):
    """Asks an LLM to turn an executed plan into one final answer."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._llm = llm
        self._system_prompt = (
            DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
        )
        self._temperature = temperature
        self._max_tokens = max_tokens

    def aggregate(self, context: ExecutionContext) -> str:
        """Return the final answer for ``context``.

        Raises:
            AggregatorError: ``context`` is not an ExecutionContext, the LLM
                call failed, or the model returned nothing.
        """
        if not isinstance(context, ExecutionContext):
            raise AggregatorError(
                "aggregate() expects an ExecutionContext, "
                f"got {type(context).__name__}."
            )

        request = LLMRequest(
            messages=[
                Message.system(self._system_prompt),
                Message.user(self._build_user_prompt(context)),
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            json_mode=False,
        )

        try:
            response: LLMResponse = self._llm.complete(request)
        except LLMError as exc:
            raise AggregatorError(f"aggregator LLM call failed: {exc}") from exc

        answer = (response.content or "").strip()
        if not answer:
            raise AggregatorError("aggregator LLM returned an empty final answer.")
        return answer

    def _build_user_prompt(self, context: ExecutionContext) -> str:
        """Lay out the goal and every step result in plan order."""
        result_by_step_id = {result.step_id: result for result in context.results}

        lines = ["Goal:", context.plan.goal, "", "Executed steps, in order:"]
        for index, step in enumerate(context.plan.steps, start=1):
            result = result_by_step_id[step.id]
            lines.extend(self._describe_step(index, step, result))
        lines.extend(
            ["", "Write the final answer for the user based on these results."]
        )
        return "\n".join(lines)

    def _describe_step(
        self, index: int, step: PlanStep, result: ExecutionResult
    ) -> list[str]:
        state = result.state
        output = (
            state.worker_output
            if state.worker_output is not None
            else state.final_result
        )
        return [
            "",
            f"Step {index}: {step.id}",
            f"  task: {step.task}",
            f"  worker: {step.worker_name}",
            f"  input: {state.worker_input or ''}",
            f"  output: {output or ''}",
        ]