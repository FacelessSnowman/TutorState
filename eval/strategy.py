"""Strategy scoring: the planner's action against Bridge's z_what_clean labels.

Strict scoring accepts only the actions the experts chose for that history.
Lenient scoring also accepts extra actions judged reasonable for the case
(how those sets are built - LLM judge, human review or both - is still open):

    acceptable = {"H_...": ["PROBE", "HINT"], ...}
"""

from __future__ import annotations

from typing import Optional, Sequence

from eval.metrics import compare_runs, score_runs
from schemas import ACTIONS


def predicted_action(row: dict) -> Optional[str]:
    planner = row.get("planner")
    return planner.get("action") if planner else None


def with_acceptable(gold: dict, acceptable: dict[str, Sequence[str]]) -> dict:
    """Copy of gold whose action lists also contain the extra acceptable actions."""
    lenient = {}
    for case_id, labels in gold.items():
        extra = [a for a in acceptable.get(case_id, ()) if a not in labels["actions"]]
        lenient[case_id] = {**labels, "actions": labels["actions"] + extra}
    return lenient


def score_strategy(results: Sequence[dict], gold: dict, acceptable: Optional[dict] = None) -> dict:
    report = {"strict": score_runs(results, gold, predicted_action, "actions", labels=ACTIONS)}
    if acceptable is not None:
        # Only any_match is meaningful here; fraction_matched and F1 mix expert and extra labels.
        report["lenient"] = score_runs(
            results, with_acceptable(gold, acceptable), predicted_action, "actions", labels=ACTIONS
        )
    return report


def compare_strategy(results: Sequence[dict], gold: dict, condition_a: str, condition_b: str) -> dict:
    return compare_runs(results, gold, predicted_action, "actions", condition_a, condition_b)
