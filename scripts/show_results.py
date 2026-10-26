"""Print a runner results file in a readable form, case by case.

    python scripts/show_results.py results/<run>.jsonl --data <the same data file the run used>
    python scripts/show_results.py results/<run>.jsonl --data <...> --judge results/<judge>.jsonl

For each case: the conversation, then each condition's diagnosis, memory
summary, planned action and tutor response, then the expert labels and
responses (BRIDGE only). Failed rows show their error instead.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.load import load_bridge_gold, load_results
from experiments.dataloader import load_cases

ORDER = ["baseline", "analyser", "memory", "full"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--data", type=Path, required=True, help="data file the run used")
    parser.add_argument("--seed", type=int, default=None, help="only show this seed")
    parser.add_argument("--judge", type=Path, default=None, help="eval.judge results output, to show ratings")
    args = parser.parse_args()

    ratings = {}
    if args.judge:
        with open(args.judge, encoding="utf-8") as f:
            for row in map(json.loads, filter(str.strip, f)):
                ratings[(row["case_id"], row["condition"], row["seed"])] = row["judge"]

    cases = {c["case_id"]: c for c in load_cases(args.data)}
    rows = json.loads(args.data.read_text(encoding="utf-8"))
    gold = load_bridge_gold(args.data) if rows and "history_id" in rows[0] else {}

    by_case = defaultdict(list)
    for row in load_results(args.results):
        if args.seed is None or row["seed"] == args.seed:
            by_case[row["case_id"]].append(row)

    for case_id, results in by_case.items():
        case = cases[case_id]
        print("=" * 100)
        print(f"{case_id}   topic: {case.get('lesson_topic')}")
        for turn in case["history"]:
            print(f"  {turn['role']:>7}: {turn['text']}")

        results.sort(key=lambda r: (ORDER.index(r["condition"]) if r["condition"] in ORDER else 99, r["seed"]))
        for r in results:
            print(f"\n  --- {r['condition']} (seed {r['seed']}) ---")
            if r.get("error"):
                print(f"  FAILED {r['error']['type']}: {r['error']['message'][:200]}")
                continue
            a, m, p = r.get("analysis"), r.get("memory"), r.get("planner")
            if a:
                print(f"  diagnosis : {a['error_type']} (correct={a['correct']}, confidence={a['confidence']})"
                      f"  misconception: {a['misconception']}")
            if m:
                print(f"  memory    : attempts={m['attempts']} correct={m['correct_attempts']} "
                      f"mastery={m['mastery']} misconception x{m['misconception_occurrences']}")
            if p:
                print(f"  action    : {p['action']} - {p['reason']}")
            print(f"  TUTOR     : {r['response']}")
            judged = ratings.get((r["case_id"], r["condition"], r["seed"]))
            if judged:
                short = {"Yes": "Y", "To some extent": "~", "No": "N"}
                labels = "  ".join(f"{dim}={short.get(v['label'], v['label'][:3])}" for dim, v in judged.items())
                print(f"  judge     : {labels}")

        if case_id in gold:
            g = gold[case_id]
            print(f"\n  --- experts ---  error types: {g['e']}  actions: {g['actions']}")
            for text in g["expert_responses"]:
                print(f"  EXPERT    : {text}")
    print("=" * 100)


if __name__ == "__main__":
    main()
