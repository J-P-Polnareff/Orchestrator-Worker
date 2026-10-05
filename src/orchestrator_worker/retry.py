"""Retry control for targeted plan execution.

A retry happens only when an evaluation completes normally, reports that the
results are not good enough, and names the steps that failed. Only those steps
are re-executed; every other step keeps the result it already has. Errors raised
while executing a step or while evaluating the plan are never turned into
retries: they propagate as they are.
"""

from __future__ import annotations


class RetryError(RuntimeError):
    """Raised when a targeted retry cannot proceed or its budget is exhausted.

    This is deliberately not an ``EvaluationError``: the evaluation itself
    completed, the verdict was simply ``passed=False``, and either no attempts
    are left or the failure could not be tied to a step. It is raised both when
    the retry budget is spent and when ``EvaluationResult.failed_step_ids`` is
    empty, because there is nothing safe to re-execute and the whole plan is
    deliberately not re-run.
    """