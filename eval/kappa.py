"""Agreement between two raters on the MRBench dimensions (judge vs human, or human vs human).

Three-level dimensions are ordinal (No < To some extent < Yes), so they use
quadratic-weighted Cohen's kappa. Revealing_of_the_Answer is collapsed to
yes/no and uses plain kappa.

    python -m eval.kappa results/judge_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from sklearn.metrics import cohen_kappa_score

THREE_LEVEL = {"No": 0, "To some extent": 1, "Yes": 2}

# Keys and label strings exactly as in MRBench (after dataset_utils renames humanlikeness).
SCALES = {
    "Mistake_Identification": THREE_LEVEL,
    "Mistake_Location": THREE_LEVEL,
    "Revealing_of_the_Answer": {
        "No": 0,
        "Yes (and the answer is correct)": 1,
        "Yes (but the answer is incorrect)": 1,
    },
    "Providing_Guidance": THREE_LEVEL,
    "Actionability": THREE_LEVEL,
    "Coherence": THREE_LEVEL,
    "Tutor_Tone": {"Offensive": 0, "Neutral": 1, "Encouraging": 2},
    "Humanlikeness": THREE_LEVEL,
}
DIMENSIONS = list(SCALES)
BINARY = {"Revealing_of_the_Answer"}


def to_score(dimension: str, label: str) -> int:
    try:
        return SCALES[dimension][label.strip()]
    except KeyError:
        raise ValueError(f"{label!r} is not a valid {dimension} label") from None


def agreement(rater_a: Sequence[str], rater_b: Sequence[str], dimension: str) -> dict:
    if len(rater_a) != len(rater_b):
        raise ValueError("both raters must label the same responses")
    a = [to_score(dimension, x) for x in rater_a]
    b = [to_score(dimension, x) for x in rater_b]
    weights = None if dimension in BINARY else "quadratic"
    # kappa is undefined (nan) when both raters only ever use one label
    return {
        "n": len(a),
        "kappa": float(cohen_kappa_score(a, b, weights=weights)),
        "exact_agreement": sum(x == y for x, y in zip(a, b)) / len(a),
    }


def agreement_table(rows: Sequence[dict], dimensions: Sequence[str] = DIMENSIONS) -> dict:
    """rows are judge output lines: {"human_ratings": {dim: label}, "judge": {dim: {"label": ...}}}."""
    table = {}
    for dim in dimensions:
        scored = [r for r in rows if dim in r.get("judge", {})]
        if scored:
            table[dim] = agreement(
                [r["human_ratings"][dim] for r in scored], [r["judge"][dim]["label"] for r in scored], dim
            )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge-vs-human agreement per dimension")
    parser.add_argument("judge_file")
    args = parser.parse_args()

    with open(args.judge_file, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    # One table per prompt version, so before/after a prompt change are never mixed.
    for version in dict.fromkeys(r.get("prompt_version") for r in rows):
        version_rows = [r for r in rows if r.get("prompt_version") == version]
        print(f"\nprompt version {version} ({len(version_rows)} responses)")
        print(f"{'dimension':<26}{'n':>5}{'kappa':>8}{'exact':>8}")
        for dim, stats in agreement_table(version_rows).items():
            print(f"{dim:<26}{stats['n']:>5}{stats['kappa']:>8.3f}{stats['exact_agreement']:>8.3f}")


if __name__ == "__main__":
    main()
