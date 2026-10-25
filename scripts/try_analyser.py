"""Run the analyser on BRIDGE *train* cases with a real LLM and compare with the experts.

    pip install anthropic
    set ANTHROPIC_API_KEY=...            (PowerShell: $env:ANTHROPIC_API_KEY="...")
    python scripts/try_analyser.py --n 20

Prints each case with the analyser's label next to the expert label(s), then the
share of cases where the analyser matched any expert. For tuning the prompt on
train only; final scores come from Track D's eval on validation/test.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.analyser import analyse
from llm import ClaudeLLM

TRAIN = Path(__file__).resolve().parent.parent / "data/Bridge/bridge_data_size_142/142_train.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="number of train cases to run")
    args = parser.parse_args()

    cases = json.loads(TRAIN.read_text(encoding="utf-8"))[: args.n]
    llm = ClaudeLLM()
    hits, confusions = 0, Counter()

    for case in cases:
        gold = sorted({a["e_clean"] for a in case["annotations"]})
        result = analyse(llm, case["c_h"], lesson_topic=case["lesson_topic"], include_reasoning=True)
        hit = result["error_type"] in gold
        hits += hit
        if not hit:
            confusions[(result["error_type"], "/".join(gold))] += 1

        print(f"=== {case['history_id']}  {'MATCH' if hit else 'miss'}  analyser={result['error_type']}  expert={gold}")
        for turn in case["c_h"]:
            print(f"  {turn['role']:>7}: {turn['text']}")
        print(f"  reasoning: {result['reasoning']}")
        print(f"  correct={result['correct']}  misconception={result['misconception']}  confidence={result['confidence']}\n")

    print(f"Matched an expert on {hits}/{len(cases)} cases ({hits / len(cases):.0%})")
    if confusions:
        print("Most common misses (analyser said -> experts said):")
        for (pred, gold), n in confusions.most_common(5):
            print(f"  {pred} -> {gold}: {n}")


if __name__ == "__main__":
    main()
