"""LLM-backed evaluator.

Flow::

    ExecutionContext -> prompt -> LLMClient.complete -> JSON -> EvaluationResult

It depends on the ``LLMClient`` abstraction only and it only reads the context:
it never executes a worker, never retries and never modifies the plan.

The prompt is built from ``context.plan.steps`` (the plan's own order) and each
result is looked up by ``step.id``, so a shuffled ``context.results`` cannot
change the order the model sees.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..context import ExecutionContext
from ..evaluation import EvaluationResult
from ..execution import ExecutionResult
from ..llm import LLMClient, LLMError, LLMRequest, LLMResponse, Message
from ..plan import PlanStep
from .base import Evaluator, EvaluatorError

DEFAULT_SYSTEM_PROMPT = (
    "You are an execution-result evaluator. You judge whether the results of a "
    "plan that has already been executed are good enough to support a final "
    "answer.\n"
    "\n"
    "Rules:\n"
    "- Judge only the plan and the step results you are given.\n"
    "- Never re-plan, re-run, retry or change the work; the work is finished.\n"
    "- Never call a worker and never modify the plan.\n"
    '- Reply with JSON only, using exactly two keys: "passed" and "reason".\n'
    '- "passed" is a JSON boolean: true when the results support a final answer.\n'
    '- "reason" is a non-empty string explaining the verdict.\n'
    "- No markdown, no code fences, no prose outside the JSON object.\n"
)


class LLMEvaluator(Evaluator):
    """Asks an LLM to judge an executed plan and parses a strict JSON verdict."""

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

    def evaluate(self, context: ExecutionContext) -> EvaluationResult:
        """Judge ``context`` and return the parsed verdict.

        Raises:
            EvaluatorError: ``context`` is not an ExecutionContext, the LLM call
                failed, or the model did not return a valid JSON verdict.
        """
        if not isinstance(context, ExecutionContext):
            raise EvaluatorError(
                "evaluate() expects an ExecutionContext, "
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
            raise EvaluatorError(f"evaluator LLM call failed: {exc}") from exc

        return self._parse_result(response.content)

    def _parse_result(self, content: str) -> EvaluationResult:
        """Parse the model's JSON verdict. Never guesses or applies defaults."""
        payload = self._load_json(content)

        if "passed" not in payload:
            raise EvaluatorError(
                "evaluator output is missing required field 'passed'."
            )
        passed = payload["passed"]
        if not isinstance(passed, bool):
            raise EvaluatorError(
                "evaluator field 'passed' must be a boolean, "
                f"got {type(passed).__name__}."
            )

        reason = self._require_reason(payload)
        return EvaluationResult(passed=passed, reason=reason)

    def _load_json(self, content: str) -> Mapping[str, Any]:
        text = (content or "").strip()
        if not text:
            raise EvaluatorError("evaluator LLM returned an empty response.")

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise EvaluatorError(
                f"evaluator LLM returned invalid JSON: {exc}"
            ) from exc

        if not isinstance(payload, Mapping):
            raise EvaluatorError(
                "evaluator output must be a JSON object, "
                f"got {type(payload).__name__}."
            )
        return payload

    def _require_reason(self, payload: Mapping[str, Any]) -> str:
        if "reason" not in payload:
            raise EvaluatorError(
                "evaluator output is missing required field 'reason'."
            )
        value = payload["reason"]
        if not isinstance(value, str):
            raise EvaluatorError(
                "evaluator field 'reason' must be a string, "
                f"got {type(value).__name__}."
            )
        reason = value.strip()
        if not reason:
            raise EvaluatorError("evaluator field 'reason' must not be empty.")
        return reason

    def _build_user_prompt(self, context: ExecutionContext) -> str:
        """Lay out the goal and every step result in plan order."""
        result_by_step_id = {result.step_id: result for result in context.results}

        lines = ["Goal:", context.plan.goal, "", "Executed steps, in plan order:"]
        for index, step in enumerate(context.plan.steps, start=1):
            result = result_by_step_id[step.id]
            lines.extend(self._describe_step(index, step, result))
        lines.extend(
            [
                "",
                "Judge whether these results are enough to produce the final "
                "answer.",
                "",
                "Reply with JSON only:",
                '{"passed": true/false, "reason": "..."}',
            ]
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
            f"Step {index}:",
            f"  id: {step.id}",
            f"  task: {step.task}",
            f"  worker: {step.worker_name}",
            f"  input: {state.worker_input or ''}",
            f"  output: {output or ''}",
        ]