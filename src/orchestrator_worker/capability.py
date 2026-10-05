"""Worker capability: a provider-neutral description of what a worker can do.

A capability is pure data: a worker name plus a short, human-readable
description that the planner can reason about. It carries no routing score, no
confidence and no model metadata, and it never talks to an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass


class CapabilityError(RuntimeError):
    """Raised when a worker capability is structurally invalid."""


def _validate_text(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise CapabilityError(f"{field} must be a string, got {type(value).__name__}.")
    if not value.strip():
        raise CapabilityError(f"{field} must be a non-empty string.")


@dataclass(frozen=True)
class WorkerCapability:
    """What a worker does, in a form the planner can reason about."""

    name: str
    description: str

    def __post_init__(self) -> None:
        _validate_text(self.name, "WorkerCapability.name")
        _validate_text(self.description, "WorkerCapability.description")