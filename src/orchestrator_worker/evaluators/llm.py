"""LLM-backed evaluator.

Flow::

    ExecutionContext -> prompt -> LLMClient.complete -> JSON -> EvaluationResult

It depends on the ``LLMClient`` abstraction only and it only reads the context:
it never executes a worker, never retries and never modifies the plan.

The prompt is built from ``context.plan.steps`` (the plan's own order) and each
result is looked up by ``step.id``, so a shuffled ``context.results`` cannot
change the order the model sees. The model must return one step evaluation per
plan step; the evaluator validates the returned step ids against the plan and
normalizes the order back to the plan order.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..context import ExecutionContext
from ..evaluation import EvaluationResult, StepEvaluation
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
    "- Evaluate the overall result, then evaluate every step individually.\n"
    "- Refer to each step by the exact step id given for it in the plan.\n"
    "- Every plan step must appear exactly once; never invent or omit a step id.\n"
    '- Reply with JSON only, using exactly three keys: "passed", "reason" and '
    '"step_evaluations".\n'
    '- "passed" is a JSON boolean: true when the results support a final answer.\n'
    '- "reason" is a non-empty string explaining the overall verdict.\n'
    '- "step_evaluations" is a JSON array with one object per plan step.\n'
    '- Every step object has exactly "step_id", "passed" and "feedback".\n'
    '- "step_id" is one of the plan step ids; "passed" is a JSON boolean.\n'
    '- "feedback" is a string: empty when the step passed, otherwise a short, '
    "actionable explanation of what is wrong.\n"
    "- An overall pass must not be reported while any step is marked as failed.\n"
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
                failed, or the model did not return a valid JSON verdict that
                covers exactly the plan's steps.
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

        return self._parse_result(response.content, context)

    def _parse_result(
        self, content: str, context: ExecutionContext
    ) -> EvaluationResult:
        """Parse the model's JSON verdict. Never guesses or applies defaults."""
        payload = self._load_json(content)

        passed = self._require_bool(payload, "passed", "evaluator output")
        reason = self._require_reason(payload)
        step_evaluations = self._parse_step_evaluations(payload, context)

        return EvaluationResult(
            passed=passed,
            reason=reason,
            step_evaluations=step_evaluations,
        )

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

    def _require_bool(self, payload: Mapping[str, Any], key: str, where: str) -> bool:
        if key not in payload:
            raise EvaluatorError(f"{where} is missing required field {key!r}.")
        value = payload[key]
        if not isinstance(value, bool):
            raise EvaluatorError(
                f"{where} field {key!r} must be a boolean, "
                f"got {type(value).__name__}."
            )
        return value

    def _require_text(
        self, payload: Mapping[str, Any], key: str, where: str, *, allow_empty: bool
    ) -> str:
        if key not in payload:
            raise EvaluatorError(f"{where} is missing required field {key!r}.")
        value = payload[key]
        if not isinstance(value, str):
            raise EvaluatorError(
                f"{where} field {key!r} must be a string, "
                f"got {type(value).__name__}."
            )
        if not allow_empty and not value.strip():
            raise EvaluatorError(f"{where} field {key!r} must not be empty.")
        return value

    def _require_reason(self, payload: Mapping[str, Any]) -> str:
        reason = self._require_text(
            payload, "reason", "evaluator output", allow_empty=False
        )
        return reason.strip()

    def _parse_step_evaluations(
        self, payload: Mapping[str, Any], context: ExecutionContext
    ) -> tuple[StepEvaluation, ...]:
        """Validate the per-step verdicts against the plan and order them.

        Unknown, missing and duplicate step ids are all rejected: the evaluator
        never silently drops, invents or completes a step.
        """
        if "step_evaluations" not in payload:
            raise EvaluatorError(
                "evaluator output is missing required field 'step_evaluations'."
            )
        raw = payload["step_evaluations"]
        if not isinstance(raw, list):
            raise EvaluatorError(
                "evaluator field 'step_evaluations' must be a JSON array, "
                f"got {type(raw).__name__}."
            )

        expected_ids = [step.id for step in context.plan.steps]
        expected = set(expected_ids)

        parsed: dict[str, StepEvaluation] = {}
        for index, entry in enumerate(raw):
            where = f"evaluator field 'step_evaluations[{index}]'"
            if not isinstance(entry, Mapping):
                raise EvaluatorError(
                    f"{where} must be a JSON object, got {type(entry).__name__}."
                )

            step_id = self._require_text(
                entry, "step_id", where, allow_empty=False
            )
            if step_id not in expected:
                raise EvaluatorError(
                    f"{where} references unknown step {step_id!r}; "
                    f"expected: {', '.join(expected_ids)}."
                )
            if step_id in parsed:
                raise EvaluatorError(
                    f"{where} repeats step id {step_id!r}; every step must "
                    "appear exactly once."
                )

            passed = self._require_bool(entry, "passed", where)
            feedback = self._require_text(
                entry, "feedback", where, allow_empty=True
            )
            parsed[step_id] = StepEvaluation(
                step_id=step_id, passed=passed, feedback=feedback
            )

        missing = [step_id for step_id in expected_ids if step_id not in parsed]
        if missing:
            raise EvaluatorError(
                "evaluator output is missing step evaluations for: "
                f"{', '.join(missing)}; every plan step must appear exactly once."
            )

        return tuple(parsed[step_id] for step_id in expected_ids)

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
                "Evaluate the overall result, then evaluate each step "
                "individually.",
                "Use exactly the step ids listed above: every step id must "
                "appear exactly once, and no other step id may be used.",
                "",
                "Reply with JSON only:",
                '{"passed": true/false, "reason": "...", "step_evaluations": ['
                '{"step_id": "...", "passed": true/false, "feedback": "..."}]}',
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