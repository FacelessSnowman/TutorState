"""Hand-scoring sheets for the human kappa sample.

export writes two CSVs: the sheet scorers fill in (shuffled, blind - it shows
no condition, seed or model name) and a separate key file mapping each
item_id back to the response. Scorers use the rubric in
prompts/judge_system.txt and type the exact label strings into the dimension
columns. Each scorer fills their own copy of the sheet.

    python -m eval.human_sheet export --results results/run.jsonl --bridge data/Bridge/bridge_data_size_142/142_test.json --n 40 --out results/human_sheet.csv
    python -m eval.human_sheet agree results/sheet_alice.csv results/sheet_bob.csv
    python -m eval.human_sheet vs-judge results/sheet_alice.csv results/human_sheet_key.csv results/judge_run.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable, Sequence

from eval.judge import RESULTS_KEY, _format_history
from eval.kappa import DIMENSIONS, agreement, to_score

TEXT_COLUMNS = ["item_id", "conversation", "reference_solution", "expert_response", "candidate_response"]


def results_items(results: Sequence[dict], contexts: dict) -> list[dict]:
    """One item per pipeline response: {"ids": {...}, context fields, "candidate_response"}."""
    return [
        {"ids": {k: r[k] for k in RESULTS_KEY}, **contexts[r["case_id"]], "candidate_response": r["response"]}
        for r in results
        if r.get("response")
    ]


def export_sheet(items: Sequence[dict], out_path, n: int, seed: int = 0) -> Path:
    """Write a random sample of n items to out_path and the key to <out>_key.csv."""
    out_path = Path(out_path)
    sample = random.Random(seed).sample(list(items), min(n, len(items)))
    key_path = out_path.with_name(out_path.stem + "_key.csv")
    id_fields = list(sample[0]["ids"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # utf-8-sig so Excel opens the files with the right encoding
    with open(out_path, "w", newline="", encoding="utf-8-sig") as sheet, \
            open(key_path, "w", newline="", encoding="utf-8-sig") as key:
        sheet_writer = csv.DictWriter(sheet, TEXT_COLUMNS + DIMENSIONS)
        key_writer = csv.DictWriter(key, ["item_id"] + id_fields)
        sheet_writer.writeheader()
        key_writer.writeheader()
        for i, item in enumerate(sample, 1):
            item_id = f"item{i:03d}"
            sheet_writer.writerow({
                "item_id": item_id,
                "conversation": _format_history(item["conversation_history"]),
                "reference_solution": item.get("ground_truth_solution") or "",
                "expert_response": item.get("expert_response") or "",
                "candidate_response": item["candidate_response"],
            })
            key_writer.writerow({"item_id": item_id, **item["ids"]})
    return key_path


def read_sheet(path) -> dict[str, dict[str, str]]:
    """{item_id: {dimension: label}} for fully scored rows; rows left blank are skipped."""
    scored = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            labels = {dim: (row.get(dim) or "").strip() for dim in DIMENSIONS}
            filled = [dim for dim, label in labels.items() if label]
            if not filled:
                continue
            if len(filled) != len(DIMENSIONS):
                raise ValueError(f"{path}: {row['item_id']} is only partly scored")
            for dim, label in labels.items():
                to_score(dim, label)  # raises on a typo
            scored[row["item_id"]] = labels
    return scored


def _agreement_on(pairs: Iterable[tuple[dict, dict]]) -> dict:
    pairs = list(pairs)
    if not pairs:
        raise ValueError("no items scored by both raters")
    return {
        dim: agreement([a[dim] for a, _ in pairs], [b[dim] for _, b in pairs], dim) for dim in DIMENSIONS
    }


def compare_sheets(path_a, path_b) -> dict:
    """Human-vs-human agreement on the items both scorers filled in."""
    a, b = read_sheet(path_a), read_sheet(path_b)
    return _agreement_on((a[i], b[i]) for i in sorted(set(a) & set(b)))


def sheet_vs_judge(sheet_path, key_path, judge_rows: Sequence[dict]) -> dict:
    """Human-vs-judge agreement, joining the sheet to judge output lines through the key."""
    human = read_sheet(sheet_path)
    with open(key_path, newline="", encoding="utf-8-sig") as f:
        key = {row.pop("item_id"): row for row in csv.DictReader(f)}

    def ids_of(row: dict, fields) -> tuple:
        return tuple(str(row[k]) for k in fields)

    fields = list(next(iter(key.values())))
    judged = {ids_of(r, fields): {d: v["label"] for d, v in r["judge"].items()} for r in judge_rows}
    pairs = [(human[i], judged[ids_of(key[i], fields)]) for i in sorted(human) if ids_of(key[i], fields) in judged]
    return _agreement_on(pairs)


def _print_table(table: dict) -> None:
    print(f"{'dimension':<26}{'n':>5}{'kappa':>8}{'exact':>8}")
    for dim, stats in table.items():
        print(f"{dim:<26}{stats['n']:>5}{stats['kappa']:>8.3f}{stats['exact_agreement']:>8.3f}")


def main() -> None:
    from eval.load import load_bridge_contexts, load_mrbench_contexts, load_results

    parser = argparse.ArgumentParser(description="Hand-scoring sheets for the human kappa sample")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--results", required=True)
    source = export.add_mutually_exclusive_group(required=True)
    source.add_argument("--bridge", help="Bridge split file the results were run on")
    source.add_argument("--mrbench-split", choices=["train", "validation", "test"])
    export.add_argument("--n", type=int, default=40)
    export.add_argument("--seed", type=int, default=0)
    export.add_argument("--out", required=True)
    agree = sub.add_parser("agree")
    agree.add_argument("sheet_a")
    agree.add_argument("sheet_b")
    vs_judge = sub.add_parser("vs-judge")
    vs_judge.add_argument("sheet")
    vs_judge.add_argument("key")
    vs_judge.add_argument("judge_file")
    args = parser.parse_args()

    if args.command == "export":
        contexts = load_bridge_contexts(args.bridge) if args.bridge else load_mrbench_contexts(args.mrbench_split)
        key_path = export_sheet(results_items(load_results(args.results), contexts), args.out, args.n, args.seed)
        print(f"wrote {args.out} and {key_path} (keep the key away from scorers)")
    elif args.command == "agree":
        _print_table(compare_sheets(args.sheet_a, args.sheet_b))
    else:
        with open(args.judge_file, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        _print_table(sheet_vs_judge(args.sheet, args.key, rows))


if __name__ == "__main__":
    main()
