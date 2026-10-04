"""Phase 4E tests: the EvaluationResult data model."""

from __future__ import annotations

import ast
import pathlib
from dataclasses import FrozenInstanceError

import pytest

from orchestrator_worker.evaluation import EvaluationError, EvaluationResult


def test_evaluation_result_stores_a_pass():
    result = EvaluationResult(passed=True, reason="the results are enough")

    assert result.passed is True
    assert result.reason == "the results are enough"


def test_evaluation_result_stores_a_failure():
    result = EvaluationResult(passed=False, reason="step_2 produced nothing")

    assert result.passed is False
    assert result.reason == "step_2 produced nothing"


def test_evaluation_result_is_frozen():
    result = EvaluationResult(passed=True, reason="ok")

    with pytest.raises(FrozenInstanceError):
        result.passed = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.reason = "other"  # type: ignore[misc]


def test_evaluation_result_is_a_dataclass_with_the_expected_fields():
    assert EvaluationResult.__dataclass_fields__.keys() == {"passed", "reason"}


@pytest.mark.parametrize("value", [1, 0, "true", "yes", None, []])
def test_passed_must_be_a_bool(value):
    with pytest.raises(EvaluationError, match="EvaluationResult.passed"):
        EvaluationResult(passed=value, reason="ok")


@pytest.mark.parametrize("value", [None, 123, True, ["reason"]])
def test_reason_must_be_a_string(value):
    with pytest.raises(EvaluationError, match="EvaluationResult.reason"):
        EvaluationResult(passed=True, reason=value)


@pytest.mark.parametrize("value", ["", "   ", "\n\t"])
def test_reason_must_not_be_empty(value):
    with pytest.raises(EvaluationError, match="non-empty"):
        EvaluationResult(passed=True, reason=value)


def test_evaluation_error_is_a_runtime_error():
    assert issubclass(EvaluationError, RuntimeError)


def test_evaluation_module_is_pure_data():
    from orchestrator_worker import evaluation as evaluation_module

    tree = ast.parse(pathlib.Path(evaluation_module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))

    assert imported == ["__future__", "dataclasses"]