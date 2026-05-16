from __future__ import annotations

from typing import Any

from ..types import EvalTask, GraderResult, GraderType


def aggregate_results(task: EvalTask, results: list[GraderResult]) -> tuple[float, bool]:
    if not results:
        return 0.0, False
    total_weight = sum(result.weight for result in results if result.weight > 0)
    weighted_score = sum(result.score * result.weight for result in results if result.weight > 0) / max(total_weight, 1e-9)
    if task.aggregation == "binary":
        passed = all(result.passed for result in results if result.weight > 0)
    elif task.aggregation == "weighted":
        passed = weighted_score >= task.threshold
    else:
        required_pass = all(result.passed for result in results if result.weight >= 1.0)
        passed = required_pass and weighted_score >= task.threshold
    return round(weighted_score, 3), passed


def _result(
    grader: str,
    grader_type: GraderType,
    method: str,
    score: float,
    passed: bool,
    weight: float,
    summary: str,
    assertions: list[dict[str, Any]],
) -> GraderResult:
    return GraderResult(grader, grader_type, method, round(score, 3), passed, weight, summary, assertions)

