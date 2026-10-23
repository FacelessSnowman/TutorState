"""Diagnosis scoring: the analyser's error_type against Bridge's e_clean labels."""

from __future__ import annotations

from typing import Optional, Sequence, get_args

from eval.metrics import compare_runs, score_runs
from schemas import ErrorType

ERROR_TYPES = list(get_args(ErrorType))


def predicted_error_type(row: dict) -> Optional[str]:
    analysis = row.get("analysis")
    return analysis.get("error_type") if analysis else None


def score_diagnosis(results: Sequence[dict], gold: dict) -> dict:
    return score_runs(results, gold, predicted_error_type, "e", labels=ERROR_TYPES)


def compare_diagnosis(results: Sequence[dict], gold: dict, condition_a: str, condition_b: str) -> dict:
    return compare_runs(results, gold, predicted_error_type, "e", condition_a, condition_b)
