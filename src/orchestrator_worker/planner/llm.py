"""LLM-backed planner.

Flow::

    task -> prompt -> LLMClient.complete -> JSON text -> parse -> validate -> Plan

It depends on the ``LLMClient`` abstraction and on a read-only view of the
available workers: their names plus the capability each one declares. It never
touches the worker registry, never executes a worker, and it never chooses a
worker itself - that decision belongs to the model, which only has to pick from
the declared capabilities.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ..capability import WorkerCapability
from ..llm import LLMClient, LLMError, LLMRequest, LLMResponse, Message
from ..plan import Plan, PlanError, PlanStep
from .base import (
    Planner,
    PlannerError,
    PlannerLLMError,
    PlanParsingError,
    PlanValidationError,
    UnknownWorkerError,
)

WORKER_PLACEHOLDER = "{available_workers}"

DEFAULT_SYSTEM_PROMPT = (
    "You are a planning component. Turn the user's task into a short, ordered "
    "plan.\n"
    "\n"
    "Available workers:\n"
    + WORKER_PLACEHOLDER
    + "\n"
    "\n"
    "Rules:\n"
    "- Reply with JSON only. No markdown, no code fences, no prose, no explanation.\n"
    '- The JSON object has exactly two keys: "goal" and "steps".\n'
    '- "goal" is a non-empty string that restates the user task.\n'
    '- "steps" is a non-empty, ordered JSON array of step objects.\n'
    '- Every step object has exactly three keys: "id", "task", "worker_name".\n'
    '- "id" is a unique non-empty string such as "step_1".\n'
    '- "task" is a non-empty string describing the work of that step.\n'
    '- "worker_name" must be one of the available workers listed above.\n'
    "- Choose the worker whose declared capability best matches the step's task.\n"
    "- Do not invent worker names and do not assume any capability beyond the "
    "ones listed above.\n"
    "- Plan only. Never execute the task and never report execution results.\n"
    "\n"
    "Example:\n"
    '{"goal": "example goal", "steps": '
    '[{"id": "step_1", "task": "example task", "worker_name": "research"}]}'
)


class LLMPlanner(Planner):
    """Asks an LLM for a JSON plan and validates the result."""

    def __init__(
        self,
        llm: LLMClient,
        available_workers: Sequence[str | WorkerCapability],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._llm = llm
        names, descriptions = _clean_workers(available_workers)
        self._available_workers = tuple(sorted(names))
        self._available = frozenset(self._available_workers)
        self._descriptions = descriptions
        self._temperature = temperature
        self._max_tokens = max_tokens

        template = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
        self._system_prompt = template.replace(
            WORKER_PLACEHOLDER, self._describe_workers()
        )

    @property
    def available_workers(self) -> tuple[str, ...]:
        """The worker names this planner is allowed to reference."""
        return self._available_workers

    def create_plan(self, task: str) -> Plan:
        """Ask the LLM for a plan and return the validated result."""
        task = (task or "").strip()
        if not task:
            raise PlannerError("task must be a non-empty string.")

        request = LLMRequest(
            messages=[Message.system(self._system_prompt), Message.user(task)],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            json_mode=True,
        )

        try:
            response: LLMResponse = self._llm.complete(request)
        except LLMError as exc:
            raise PlannerLLMError(f"planner LLM call failed: {exc}") from exc

        return self._parse_plan(response.content)

    def _describe_workers(self) -> str:
        """Bullet list used inside the prompt.

        Each worker is one bullet; when the planner was given a
        :class:`WorkerCapability`, its description follows on the next line so
        the model sees what the worker actually declares.
        """
        if not self._available_workers:
            return "- (none)"
        lines: list[str] = []
        for name in self._available_workers:
            lines.append(f"- {name}")
            description = self._descriptions.get(name)
            if description:
                lines.append(f"  Capability: {description}")
        return "\n".join(lines)

    def _worker_summary(self) -> str:
        """Inline list used inside error messages."""
        if not self._available_workers:
            return "(none)"
        return ", ".join(self._available_workers)

    def _parse_plan(self, content: str) -> Plan:
        payload = self._load_json(content)

        goal = self._require_text(payload, "goal", "plan")

        if "steps" not in payload:
            raise PlanValidationError("plan is missing required field 'steps'.")
        raw_steps = payload["steps"]
        if not isinstance(raw_steps, list):
            raise PlanValidationError(
                f"plan field 'steps' must be a JSON array, "
                f"got {type(raw_steps).__name__}."
            )
        if not raw_steps:
            raise PlanValidationError("plan field 'steps' must not be empty.")

        seen_ids: set[str] = set()
        steps = [
            self._parse_step(raw, index, seen_ids)
            for index, raw in enumerate(raw_steps)
        ]

        try:
            return Plan(goal=goal, steps=steps)
        except PlanError as exc:
            raise PlanValidationError(str(exc)) from exc

    def _load_json(self, content: str) -> Mapping[str, Any]:
        text = (content or "").strip()
        if not text:
            raise PlanParsingError("planner LLM returned an empty response.")

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PlanParsingError(
                f"planner LLM returned invalid JSON: {exc}"
            ) from exc

        if not isinstance(payload, Mapping):
            raise PlanParsingError(
                f"planner output must be a JSON object, "
                f"got {type(payload).__name__}."
            )
        return payload

    def _parse_step(self, raw: Any, index: int, seen_ids: set[str]) -> PlanStep:
        where = f"step {index}"
        if not isinstance(raw, Mapping):
            raise PlanValidationError(
                f"{where} must be a JSON object, got {type(raw).__name__}."
            )

        step_id = self._require_text(raw, "id", where)
        task = self._require_text(raw, "task", where)
        worker_name = self._require_text(raw, "worker_name", where)

        if step_id in seen_ids:
            raise PlanValidationError(f"{where} reuses duplicate step id {step_id!r}.")
        seen_ids.add(step_id)

        if worker_name not in self._available:
            raise UnknownWorkerError(
                f"planner generated unknown worker {worker_name!r}; "
                f"available workers: {self._worker_summary()}"
            )

        return PlanStep(id=step_id, task=task, worker_name=worker_name)

    def _require_text(self, payload: Mapping[str, Any], key: str, where: str) -> str:
        if key not in payload:
            raise PlanValidationError(f"{where} is missing required field {key!r}.")
        value = payload[key]
        if not isinstance(value, str):
            raise PlanValidationError(
                f"{where} field {key!r} must be a string, "
                f"got {type(value).__name__}."
            )
        text = value.strip()
        if not text:
            raise PlanValidationError(f"{where} field {key!r} must not be empty.")
        return text


def _clean_workers(
    workers: Sequence[str | WorkerCapability],
) -> tuple[set[str], dict[str, str]]:
    """Split worker names from optional capability descriptions.

    Plain names stay name-only; a :class:`WorkerCapability` contributes both its
    name and its description. Duplicate names collapse to one entry.
    """
    names: set[str] = set()
    descriptions: dict[str, str] = {}
    for worker in workers:
        if isinstance(worker, WorkerCapability):
            names.add(worker.name)
            descriptions[worker.name] = worker.description
        elif isinstance(worker, str) and worker.strip():
            names.add(worker.strip())
        else:
            raise ValueError(
                "available workers must be non-empty strings or "
                f"WorkerCapability objects, got {worker!r}."
            )
    return names, descriptions