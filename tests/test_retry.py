"""Phase 4E tests: the RetryError contract."""

from __future__ import annotations

import ast
import pathlib

from orchestrator_worker.evaluation import EvaluationError
from orchestrator_worker.retry import RetryError


def test_retry_error_is_a_runtime_error():
    assert issubclass(RetryError, RuntimeError)


def test_retry_error_is_not_an_evaluation_error():
    """An exhausted retry budget is not a broken evaluation."""
    assert not issubclass(RetryError, EvaluationError)


def test_retry_error_carries_a_message():
    error = RetryError("evaluation failed after 2 attempt(s): not enough")
    assert "evaluation failed after 2 attempt(s)" in str(error)
    assert "not enough" in str(error)


def test_retry_module_has_no_project_dependencies():
    from orchestrator_worker import retry as retry_module

    tree = ast.parse(pathlib.Path(retry_module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))

    assert imported == ["__future__"]