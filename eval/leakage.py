"""Answer-leakage check: does a tutor response contain the problem's final answer?

Only works where a reference solution exists (MRBench). The final answer is
the last number on the last line of Ground_Truth_Solution. Validated against
MRBench's human Revealing_of_the_Answer label:

    python -m eval.leakage --split train
"""

from __future__ import annotations

import argparse
import math
import re
from typing import Optional, Sequence

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    values = []
    for match in NUMBER.findall(text):
        try:
            values.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return values


def final_answer(solution: Optional[str]) -> Optional[float]:
    if not solution:
        return None
    lines = [line for line in solution.splitlines() if line.strip()]
    numbers = numbers_in(lines[-1]) if lines else []
    return numbers[-1] if numbers else None


def _contains(numbers: list[float], answer: float) -> bool:
    return any(math.isclose(n, answer) for n in numbers)


def reveals_answer(response: str, solution: Optional[str], history: Optional[str] = None) -> Optional[bool]:
    """True if the response mentions the final answer.

    None when it cannot be checked: there is no reference answer, or the
    answer already appears in the conversation (in the problem text or said by
    the student), so repeating it is not evidence of revealing. Those cases
    are left to the LLM judge's Revealing_of_the_Answer rating.
    """
    answer = final_answer(solution)
    if answer is None or (history and _contains(numbers_in(history), answer)):
        return None
    return _contains(numbers_in(response), answer)


def human_reveals(label: str) -> bool:
    """Both "Yes (and the answer is correct)" and "Yes (but ... incorrect)" count as revealing."""
    return label.strip().startswith("Yes")


def evaluate_on_mrbench(examples: Sequence[dict]) -> dict:
    """Precision/recall of reveals_answer against the human label (positives are rare)."""
    tp = fp = fn = tn = skipped = 0
    false_pos, false_neg = [], []
    for ex in examples:
        flagged = reveals_answer(ex["candidate_response"], ex["ground_truth_solution"], ex["conversation_history"])
        if flagged is None:
            skipped += 1
            continue
        human = human_reveals(ex["human_ratings"]["Revealing_of_the_Answer"])
        key = (ex["conversation_id"], ex["response_source"])
        if flagged and human:
            tp += 1
        elif flagged:
            fp += 1
            false_pos.append(key)
        elif human:
            fn += 1
            false_neg.append(key)
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "skipped": skipped,
        "precision": precision, "recall": recall, "f1": f1,
        "false_positives": false_pos, "false_negatives": false_neg,
    }


def leakage_by_condition(results: Sequence[dict], contexts: dict) -> dict:
    """Share of checkable responses that reveal the answer, per condition.

    contexts comes from load_mrbench_contexts. Bridge has no reference answer,
    so for Bridge every response is unchecked; use the judge's
    Revealing_of_the_Answer rating there instead.
    """
    report = {}
    for row in results:
        context = contexts[row["case_id"]]
        flagged = reveals_answer(
            row.get("response") or "", context["ground_truth_solution"], str(context["conversation_history"])
        )
        stats = report.setdefault(row["condition"], {"checked": 0, "flagged": 0, "unchecked": 0})
        if flagged is None:
            stats["unchecked"] += 1
        else:
            stats["checked"] += 1
            stats["flagged"] += flagged
    for stats in report.values():
        stats["rate"] = stats["flagged"] / stats["checked"] if stats["checked"] else None
    return report


def main() -> None:
    from eval.load import load_mrbench_examples

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    args = parser.parse_args()

    report = evaluate_on_mrbench(load_mrbench_examples(args.split))
    print(f"MRBench {args.split}: tp={report['tp']} fp={report['fp']} fn={report['fn']} "
          f"tn={report['tn']} skipped={report['skipped']}")
    print(f"precision={report['precision']:.3f} recall={report['recall']:.3f} f1={report['f1']:.3f}")
    print("first false positives:", report["false_positives"][:5])
    print("first false negatives:", report["false_negatives"][:5])


if __name__ == "__main__":
    main()
