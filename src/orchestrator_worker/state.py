"""State object flowing through the Orchestrator-Worker pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class AgentState:
    """Mutable state for one orchestration run.

    Flow::

        user_task -> worker_input -> worker_output -> final_result

    The orchestrator owns the lifecycle: it fills ``worker_input`` before the
    worker runs and stores the returned text in ``worker_output``.
    """

    user_task: str
    worker_name: str | None = None
    worker_input: str | None = None
    worker_output: str | None = None
    final_result: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view, useful for logging and API output."""
        return asdict(self)