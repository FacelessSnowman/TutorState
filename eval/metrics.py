"""Multi-reference classification metrics shared by diagnosis.py and strategy.py.

A Bridge history can have several expert annotations that disagree, so every
gold value is a list of labels (one per annotation):
  * any_match: the prediction equals at least one expert's label
  * fraction_matched: share of the expert annotations the prediction agrees with
  * macro_f1 / confusion: single-label metrics, using the matched label as gold
    when the prediction matched one, and the first annotation's label otherwise

Macro-F1 is averaged only over labels that occur in gold, so a class that is
absent from the split (e.g. no_error in Bridge-142 test) does not count as 0.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from typing import Callable, Optional, Sequence

from sklearn.metrics import confusion_matrix, f1_score

PredFn = Callable[[dict], Optional[str]]


def any_match(pred: str, refs: Sequence[str]) -> bool:
    return pred in refs


def fraction_matched(pred: str, refs: Sequence[str]) -> float:
    return sum(r == pred for r in refs) / len(refs)


def resolve_gold(pred: str, refs: Sequence[str]) -> str:
    return pred if pred in refs else refs[0]


def score(preds: Sequence[str], refs: Sequence[Sequence[str]], labels: Sequence[str] = ()) -> dict:
    if len(preds) != len(refs):
        raise ValueError(f"{len(preds)} predictions but {len(refs)} gold entries")
    if not preds:
        raise ValueError("nothing to score")
    gold = [resolve_gold(p, r) for p, r in zip(preds, refs)]
    present = sorted(set(gold))
    order = list(labels) + sorted((set(gold) | set(preds)) - set(labels))
    per_class = f1_score(gold, preds, labels=present, average=None, zero_division=0)
    return {
        "n": len(preds),
        "any_match": statistics.fmean(any_match(p, r) for p, r in zip(preds, refs)),
        "fraction_matched": statistics.fmean(fraction_matched(p, r) for p, r in zip(preds, refs)),
        "macro_f1": float(per_class.mean()),
        "per_class_f1": {label: float(f) for label, f in zip(present, per_class)},
        "labels": order,
        # rows = gold label, columns = predicted label, both in `labels` order
        "confusion": confusion_matrix(gold, preds, labels=order).tolist(),
    }


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI of the mean, resampling cases with replacement."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]


def paired_difference(a: Sequence[float], b: Sequence[float], n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean of a - b over the same cases, with a paired bootstrap CI.

    The difference is significant at 5% when the CI does not contain 0.
    """
    if len(a) != len(b):
        raise ValueError("paired comparison needs the same cases on both sides")
    diffs = [x - y for x, y in zip(a, b)]
    return {"n": len(diffs), "diff": statistics.fmean(diffs), "ci": bootstrap_ci(diffs, n_boot, seed)}


def _usable_rows(results: Sequence[dict], gold: dict, pred_fn: PredFn) -> list[tuple[dict, str]]:
    missing = sorted({r["case_id"] for r in results} - set(gold))
    if missing:
        raise ValueError(f"{len(missing)} case_ids have no gold labels, e.g. {missing[:3]}")
    # Conditions without the relevant component (e.g. no planner) return None and are skipped.
    return [(r, p) for r in results if (p := pred_fn(r)) is not None]


def score_runs(
    results: Sequence[dict], gold: dict, pred_fn: PredFn, gold_key: str, labels: Sequence[str] = ()
) -> dict:
    """Score every condition: per-seed metrics, mean/std over seeds, and a case-level CI.

    Returns {condition: {...}}; conditions with no predictions are left out.
    """
    by_condition: dict = defaultdict(lambda: defaultdict(list))
    for row, pred in _usable_rows(results, gold, pred_fn):
        by_condition[row["condition"]][row["seed"]].append((row["case_id"], pred))

    report = {}
    for condition, seeds in by_condition.items():
        per_seed = {}
        for seed, pairs in sorted(seeds.items()):
            preds = [p for _, p in pairs]
            refs = [gold[case_id][gold_key] for case_id, _ in pairs]
            per_seed[seed] = score(preds, refs, labels)

        summary = {"seeds": list(per_seed), "per_seed": per_seed}
        for metric in ("any_match", "fraction_matched", "macro_f1"):
            values = [s[metric] for s in per_seed.values()]
            summary[metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        case_scores = _case_any_match(seeds, gold, gold_key)
        summary["n_cases"] = len(case_scores)
        summary["any_match"]["ci"] = bootstrap_ci(list(case_scores.values()))
        report[condition] = summary
    return report


def _case_any_match(seeds: dict, gold: dict, gold_key: str) -> dict[str, float]:
    """any_match per case, averaged over seeds (so each case counts once in the CI)."""
    hits = defaultdict(list)
    for pairs in seeds.values():
        for case_id, pred in pairs:
            hits[case_id].append(any_match(pred, gold[case_id][gold_key]))
    return {case_id: statistics.fmean(h) for case_id, h in hits.items()}


def compare_runs(
    results: Sequence[dict], gold: dict, pred_fn: PredFn, gold_key: str, condition_a: str, condition_b: str
) -> dict:
    """Paired any_match difference (a - b) over the cases both conditions scored."""
    by_condition: dict = defaultdict(lambda: defaultdict(list))
    for row, pred in _usable_rows(results, gold, pred_fn):
        by_condition[row["condition"]][row["seed"]].append((row["case_id"], pred))
    for condition in (condition_a, condition_b):
        if condition not in by_condition:
            raise ValueError(f"no predictions for condition {condition!r}")
    a = _case_any_match(by_condition[condition_a], gold, gold_key)
    b = _case_any_match(by_condition[condition_b], gold, gold_key)
    shared = sorted(set(a) & set(b))
    return paired_difference([a[c] for c in shared], [b[c] for c in shared])
