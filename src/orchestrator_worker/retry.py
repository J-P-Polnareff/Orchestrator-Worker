"""Retry control for whole-plan execution.

A retry happens only when an evaluation completes normally and reports that the
results are not good enough. Errors raised while executing the plan or while
evaluating it are never turned into retries: they propagate as they are.
"""

from __future__ import annotations


class RetryError(RuntimeError):
    """Raised when evaluation still fails after the retry budget is exhausted.

    This is deliberately not an ``EvaluationError``: the evaluation itself
    completed, the verdict was simply ``passed=False``, and no attempts are
    left. It is also not tied to any single plan step, because a retry always
    re-executes the whole plan.
    """