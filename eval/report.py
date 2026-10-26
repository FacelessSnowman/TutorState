"""One-command evaluation report for a results file from Track E's runner.

Bridge results: diagnosis, strategy (strict, and lenient if --acceptable is
given), confusion matrices and paired comparisons between conditions.
MRBench results: answer leakage per condition, and per planned action.
Either, with --judge (run_judge_results output): judge scores per condition,
mistake dimensions on mistake cases only (Bridge), paired differences per
dimension, and how often each planned action reveals the answer.

    python -m eval.report --results results/run.jsonl --bridge data/Bridge/bridge_data_size_142/142_test.json --out-dir results/report
    python -m eval.report --results results/mrbench_run.jsonl --mrbench-split test --judge results/judge_run.jsonl

Conditions are compared in the order given by --order (default: order of
first appearance), each against the one before it, matching the ablation
table: baseline -> + analyser -> + memory -> + planner.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from eval.diagnosis import compare_diagnosis, score_diagnosis
from eval.kappa import BINARY, DIMENSIONS, MISTAKE_DIMENSIONS, REPORTED_DIMENSIONS, to_score
from eval.leakage import leakage_by_condition, planned_action
from eval.load import failure_rate
from eval.metrics import bootstrap_ci, paired_difference
from eval.strategy import compare_strategy, score_strategy


def mistake_cases(gold: dict) -> set[str]:
    """Bridge histories where the student made an error: no expert chose AFFIRM
    and no expert labelled the turn no_error. MRBench's mistake dimensions only
    make sense on these; a tutor that correctly affirms a right answer would
    otherwise score low on them."""
    return {
        case_id for case_id, labels in gold.items()
        if "AFFIRM" not in labels["actions"] and "no_error" not in labels["e"]
    }


def _judge_case_scores(
    judge_rows: Sequence[dict], dimensions: Sequence[str], cases: Optional[set] = None
) -> dict:
    """{condition: {dimension: {case_id: score averaged over seeds}}}."""
    per_case: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in judge_rows:
        if cases is not None and row["case_id"] not in cases:
            continue
        for dim in dimensions:
            if dim in row["judge"]:
                per_case[row["condition"]][dim][row["case_id"]].append(to_score(dim, row["judge"][dim]["label"]))
    return {
        condition: {dim: {c: statistics.fmean(v) for c, v in by_case.items()} for dim, by_case in dims.items()}
        for condition, dims in per_case.items()
    }


def judge_summary(
    judge_rows: Sequence[dict], dimensions: Sequence[str] = DIMENSIONS, cases: Optional[set] = None
) -> dict:
    """Mean judge score per condition and dimension, with a case-level bootstrap CI.

    Scores use the kappa.SCALES numbers (No=0, To some extent=1, Yes=2;
    Revealing_of_the_Answer 0/1). Seeds are averaged per case first.
    cases restricts the summary to those case_ids (e.g. mistake_cases(gold)).
    """
    summary = {}
    for condition, dims in _judge_case_scores(judge_rows, dimensions, cases).items():
        summary[condition] = {}
        for dim, by_case in dims.items():
            values = list(by_case.values())
            summary[condition][dim] = {"n": len(values), "mean": statistics.fmean(values), "ci": bootstrap_ci(values)}
    return summary


def compare_judge(
    judge_rows: Sequence[dict], dimension: str, condition_a: str, condition_b: str, cases: Optional[set] = None
) -> dict:
    """Paired difference in a judge dimension (a - b) over the cases both conditions were judged on."""
    scores = _judge_case_scores(judge_rows, [dimension], cases)
    for condition in (condition_a, condition_b):
        if dimension not in scores.get(condition, {}):
            raise ValueError(f"no {dimension} scores for condition {condition!r}")
    a, b = scores[condition_a][dimension], scores[condition_b][dimension]
    shared = sorted(set(a) & set(b))
    return paired_difference([a[c] for c in shared], [b[c] for c in shared])


def judge_by_action(
    judge_rows: Sequence[dict], results: Sequence[dict], dimension: str = "Revealing_of_the_Answer"
) -> dict:
    """Mean judge score per (condition, planned action), e.g. how often each action reveals the answer.

    Joined to the results file by case/condition/seed; responses from conditions
    without a planner are grouped under "none".
    """
    action = {(r["case_id"], r["condition"], r["seed"]): planned_action(r) for r in results}
    grouped: dict = defaultdict(lambda: defaultdict(list))
    for row in judge_rows:
        key = (row["case_id"], row["condition"], row["seed"])
        if key in action and dimension in row["judge"]:
            grouped[row["condition"]][action[key]].append(to_score(dimension, row["judge"][dimension]["label"]))
    return {
        condition: {a: {"n": len(v), "mean": statistics.fmean(v)} for a, v in by_action.items()}
        for condition, by_action in grouped.items()
    }


def plot_confusion(result: dict, title: str, path: Path) -> None:
    """Row-normalised confusion matrix (rows = expert label, columns = predicted)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, counts = result["labels"], result["confusion"]
    rates = [[c / sum(row) if sum(row) else 0.0 for c in row] for row in counts]
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2, 1.0 * len(labels) + 1.5))
    image = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Expert (matched label)")
    ax.set_title(title)
    for i, row in enumerate(counts):
        for j, count in enumerate(row):
            if count:
                ax.text(j, i, str(count), ha="center", va="center",
                        color="white" if rates[i][j] > 0.5 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="Row-normalised proportion")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _summed_confusion(condition_report: dict) -> dict:
    """Confusion matrix summed over seeds (all seeds share the same label order)."""
    seeds = list(condition_report["per_seed"].values())
    labels = seeds[0]["labels"]
    total = [[0] * len(labels) for _ in labels]
    for s in seeds:
        for i, row in enumerate(s["confusion"]):
            for j, count in enumerate(row):
                total[i][j] += count
    return {"labels": labels, "confusion": total}


def _fmt_ci(ci) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _print_scores(title: str, report: dict, order: Sequence[str]) -> None:
    print(f"\n== {title} ==")
    if not report:
        print("(nothing to score: no condition in this file produced this output, e.g. baseline only)")
        return
    print(f"{'condition':<16}{'cases':>6}{'seeds':>6}{'any_match':>11}{'95% CI':>18}{'frac':>8}{'macroF1':>9}")
    for condition in order:
        if condition not in report:
            continue
        r = report[condition]
        print(f"{condition:<16}{r['n_cases']:>6}{len(r['seeds']):>6}"
              f"{r['any_match']['mean']:>11.3f}{_fmt_ci(r['any_match']['ci']):>18}"
              f"{r['fraction_matched']['mean']:>8.3f}{r['macro_f1']['mean']:>9.3f}")


def _print_comparisons(title: str, compare, results, gold, order: Sequence[str], scored: dict) -> dict:
    present = [c for c in order if c in scored]
    out = {}
    if len(present) < 2:
        return out
    print(f"\n-- {title}: paired any_match difference (later - earlier) --")
    for earlier, later in zip(present, present[1:]):
        diff = compare(results, gold, later, earlier)
        out[f"{later} - {earlier}"] = diff
        marker = "*" if diff["ci"][0] > 0 or diff["ci"][1] < 0 else " "
        print(f"{later + ' - ' + earlier:<34}{diff['diff']:>+8.3f} {_fmt_ci(diff['ci'])} {marker}")
    print("(* = 95% CI excludes 0)")
    return out


def _fmt_rate(rate: Optional[float]) -> str:
    return f"{rate:.3f}" if rate is not None else "-"


def _rank(order: Sequence[str], condition: str) -> int:
    return order.index(condition) if condition in order else len(order)


def _nest(by_pair: dict) -> dict:
    """{(condition, action): stats} -> {condition: {action: stats}}, so it can be saved as JSON."""
    nested: dict = defaultdict(dict)
    for (condition, action), stats in by_pair.items():
        nested[condition][action] = stats
    return dict(nested)


def _report_judge(
    report: dict, judge_rows: Sequence[dict], results: Sequence[dict], order: Sequence[str],
    gold: Optional[dict], has_planner: bool,
) -> None:
    report["judge"] = judge_summary(judge_rows)
    present = [c for c in order if c in report["judge"]]
    print("\n== Judge scores (mean; 0-2 scale, Revealing_of_the_Answer 0-1) ==")
    print(f"{'dimension':<34}" + "".join(f"{c[:12]:>13}" for c in present))
    for dim in DIMENSIONS:
        cells = [report["judge"][c].get(dim) for c in present]
        if any(cells):
            label = dim + (" (0-1)" if dim in BINARY else "") + ("" if dim in REPORTED_DIMENSIONS else " [not reported]")
            print(f"{label:<34}" + "".join(f"{c['mean']:>13.3f}" if c else f"{'-':>13}" for c in cells))

    # Mistake dimensions on Bridge: only histories where the student made an error.
    mistakes = mistake_cases(gold) if gold is not None else None
    if mistakes is not None:
        judged = {r["case_id"] for r in judge_rows}
        report["judge_mistake_cases"] = judge_summary(judge_rows, MISTAKE_DIMENSIONS, mistakes)
        print(f"\n-- Mistake dimensions on mistake cases only "
              f"({len(judged & mistakes)} of {len(judged)} judged histories; no expert AFFIRM or no_error) --")
        for dim in MISTAKE_DIMENSIONS:
            cells = [report["judge_mistake_cases"].get(c, {}).get(dim) for c in present]
            print(f"{dim:<34}" + "".join(f"{c['mean']:>13.3f}" if c else f"{'-':>13}" for c in cells))

    if len(present) >= 2:
        report["judge_comparisons"] = {}
        print("\n-- Judge: paired difference per dimension (later - earlier; * = 95% CI excludes 0) --")
        for earlier, later in zip(present, present[1:]):
            name = f"{later} - {earlier}"
            report["judge_comparisons"][name] = {}
            print(name)
            for dim in REPORTED_DIMENSIONS:
                cases = mistakes if dim in MISTAKE_DIMENSIONS else None
                try:
                    diff = compare_judge(judge_rows, dim, later, earlier, cases)
                except ValueError:
                    continue
                if not diff["n"]:
                    continue
                report["judge_comparisons"][name][dim] = diff
                marker = "*" if diff["ci"][0] > 0 or diff["ci"][1] < 0 else " "
                print(f"  {dim:<32}{diff['diff']:>+8.3f} {_fmt_ci(diff['ci'])} {marker}")

    if has_planner:
        report["judge_revealing_by_action"] = judge_by_action(judge_rows, results)
        print("\n-- Judge: revealed the answer, by planned action (share; CORRECT reveals by design) --")
        for condition in present:
            for action, s in sorted(report["judge_revealing_by_action"].get(condition, {}).items()):
                print(f"{condition:<16}{action:<12}{s['n']:>6}{s['mean']:>8.3f}")


def build_report(
    results: Sequence[dict],
    order: Sequence[str],
    gold: Optional[dict] = None,
    acceptable: Optional[dict] = None,
    contexts: Optional[dict] = None,
    judge_rows: Optional[Sequence[dict]] = None,
    out_dir: Optional[Path] = None,
) -> dict:
    report: dict = {}
    report["failures"] = failure_rate(results)
    print("\n== Failed cases (after retries; excluded from every score below) ==")
    print(f"{'condition':<16}{'rows':>6}{'failed':>8}{'rate':>8}  by type")
    for condition in order:
        s = report["failures"].get(condition)
        if s:
            print(f"{condition:<16}{s['n']:>6}{s['failed']:>8}{s['rate']:>8.1%}  {s['by_type'] or '-'}")

    if gold is not None:
        report["diagnosis"] = score_diagnosis(results, gold)
        report["strategy"] = score_strategy(results, gold, acceptable)
        _print_scores("Diagnosis (error_type vs e_clean)", report["diagnosis"], order)
        report["diagnosis_comparisons"] = _print_comparisons(
            "Diagnosis", compare_diagnosis, results, gold, order, report["diagnosis"])
        _print_scores("Strategy, strict (action vs z_what_clean)", report["strategy"]["strict"], order)
        report["strategy_comparisons"] = _print_comparisons(
            "Strategy", compare_strategy, results, gold, order, report["strategy"]["strict"])
        if "lenient" in report["strategy"]:
            _print_scores("Strategy, lenient (expert + acceptable actions; any_match only)",
                          report["strategy"]["lenient"], order)
        if out_dir is not None:
            for name, scores in (("diagnosis", report["diagnosis"]), ("strategy", report["strategy"]["strict"])):
                for condition, r in scores.items():
                    plot_confusion(_summed_confusion(r), f"{name}: {condition}", out_dir / f"confusion_{name}_{condition}.png")

    has_planner = any(r.get("planner") for r in results)

    if contexts is not None:
        report["leakage"] = leakage_by_condition(results, contexts)
        print("\n== Answer leakage (automatic check) ==")
        print(f"{'condition':<16}{'checked':>8}{'flagged':>8}{'rate':>8}{'unchecked':>10}")
        for condition in order:
            s = report["leakage"].get(condition)
            if s:
                print(f"{condition:<16}{s['checked']:>8}{s['flagged']:>8}{_fmt_rate(s['rate']):>8}{s['unchecked']:>10}")
        if has_planner:
            by_action = leakage_by_condition(results, contexts, by_action=True)
            report["leakage_by_action"] = _nest(by_action)
            print("\n-- Answer leakage by planned action (leaks under CORRECT are expected) --")
            print(f"{'condition':<16}{'action':<12}{'checked':>8}{'flagged':>8}{'rate':>8}")
            for (condition, action), s in sorted(by_action.items(), key=lambda kv: (_rank(order, kv[0][0]), kv[0][1])):
                print(f"{condition:<16}{action:<12}{s['checked']:>8}{s['flagged']:>8}{_fmt_rate(s['rate']):>8}")

    if judge_rows:
        _report_judge(report, judge_rows, results, order, gold, has_planner)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=list), encoding="utf-8")
    return report


def main() -> None:
    from eval.acceptable import load_acceptable
    from eval.load import load_bridge_gold, load_mrbench_contexts, load_results

    parser = argparse.ArgumentParser(description="Track D evaluation report")
    parser.add_argument("--results", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bridge", help="Bridge split file the results were run on")
    source.add_argument("--mrbench-split", choices=["train", "validation", "test"])
    parser.add_argument("--dataset", help="only use results rows with this dataset value, e.g. bridge142")
    parser.add_argument("--acceptable", help="acceptable-action sets: eval.acceptable output (.jsonl) or a JSON dict")
    parser.add_argument("--judge", help="run_judge_results output (JSONL)")
    parser.add_argument("--order", help="comma-separated condition order, e.g. baseline,analyser,memory,full")
    parser.add_argument("--out-dir", help="save report.json and confusion matrix PNGs here")
    args = parser.parse_args()

    results = load_results(args.results)
    if args.dataset:
        results = [r for r in results if r.get("dataset") == args.dataset]
        if not results:
            raise SystemExit(f"no results rows with dataset={args.dataset!r}")
    order = args.order.split(",") if args.order else list(dict.fromkeys(r["condition"] for r in results))
    acceptable = load_acceptable(args.acceptable) if args.acceptable else None
    judge_rows = None
    if args.judge:
        with open(args.judge, encoding="utf-8") as f:
            judge_rows = [json.loads(line) for line in f if line.strip()]

    build_report(
        results,
        order,
        gold=load_bridge_gold(args.bridge) if args.bridge else None,
        acceptable=acceptable,
        contexts=load_mrbench_contexts(args.mrbench_split) if args.mrbench_split else None,
        judge_rows=judge_rows,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )


if __name__ == "__main__":
    main()
