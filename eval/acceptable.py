"""Acceptable-action sets for lenient strategy scoring.

Our prof pointed out that more than one action can be a good next move, so
strict scoring (only the expert's action counts) undercounts reasonable plans.
Here an LLM looks at each Bridge conversation and marks each of the 7 actions
as acceptable or not for the next tutor turn.

Fairness rules:
  * Sets are built once per case, not per prediction, so every condition is
    scored against the same sets.
  * The LLM sees only what the planner sees (conversation + lesson topic):
    never the expert's action or response, and never any condition's
    prediction, so it cannot copy or favour one of them.
  * Expert actions are always accepted on top (strategy.with_acceptable).
  * A human checks a sample before lenient numbers are reported
    (export_review_sheet, then review_agreement).

    build_acceptable(llm, load_bridge_cases(path), "results/acceptable_validation.jsonl")
    acceptable = load_acceptable("results/acceptable_validation.jsonl")
    score_strategy(results, gold, acceptable)

    python -m eval.acceptable build --bridge <split file> --dry-run       # costs API credit without --dry-run
    python -m eval.acceptable summary results/acceptable_validation.jsonl --bridge <split file>
    python -m eval.acceptable export results/acceptable_validation.jsonl --bridge <split file> --n 30 --out results/acceptable_review.csv
    python -m eval.acceptable review results/acceptable_review.csv results/acceptable_validation.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from sklearn.metrics import cohen_kappa_score

from agents.prompting import build_user_prompt, format_action_definitions, load_prompt
from eval.cache import run_cached
from llm import LLM, LLMError
from schemas import ACTIONS, Turn

ACCEPTABLE_SCHEMA = {
    "type": "object",
    "properties": {
        action: {
            "type": "object",
            "properties": {"reasoning": {"type": "string"}, "acceptable": {"type": "boolean"}},
            "required": ["reasoning", "acceptable"],
            "additionalProperties": False,
        }
        for action in ACTIONS
    },
    "required": ACTIONS,
    "additionalProperties": False,
}


class AcceptableError(RuntimeError):
    def __init__(self, message: str, raw_outputs: list[str]):
        super().__init__(message)
        self.raw_outputs = raw_outputs


def system_prompt() -> str:
    return load_prompt("acceptable_actions_system.txt").format(action_definitions=format_action_definitions())


def prompt_version() -> str:
    """Short hash of the prompt; editing it invalidates the cache."""
    return hashlib.sha1(system_prompt().encode("utf-8")).hexdigest()[:8]


def load_bridge_cases(path) -> list[dict]:
    """Only the fields the LLM may see: case_id, history (role/text) and lesson topic."""
    return [
        {
            "case_id": row["history_id"],
            "history": [{"role": t["role"], "text": t["text"]} for t in row["c_h"]],
            "lesson_topic": row.get("lesson_topic"),
        }
        for row in json.loads(Path(path).read_text(encoding="utf-8"))
    ]


def parse_acceptable_output(raw: str) -> dict:
    """Validate the model's JSON. Raises ValueError with the reason it is invalid."""
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("output is not a JSON object")
    per_action = {}
    for action in ACTIONS:
        entry = data.get(action)
        if not isinstance(entry, dict) or not isinstance(entry.get("acceptable"), bool):
            raise ValueError(f"{action} is missing or has no true/false 'acceptable'")
        per_action[action] = {"reasoning": str(entry.get("reasoning", "")).strip(), "acceptable": entry["acceptable"]}
    return {
        "acceptable": [a for a in ACTIONS if per_action[a]["acceptable"]],
        "per_action": per_action,
    }


def judge_acceptable(
    llm: LLM, history: Sequence[Turn], lesson_topic: Optional[str] = None, max_attempts: int = 2
) -> dict:
    user = build_user_prompt(history, lesson_topic=lesson_topic)
    raw_outputs = []
    for _ in range(max_attempts):
        raw = llm.complete(system_prompt(), user, json_schema=ACCEPTABLE_SCHEMA)
        raw_outputs.append(raw)
        try:
            return parse_acceptable_output(raw)
        except (ValueError, json.JSONDecodeError):
            continue
    raise AcceptableError(f"no valid action judgements after {max_attempts} attempts", raw_outputs)


def build_acceptable(llm: LLM, cases: Sequence[dict], out_path) -> int:
    """Judge every case (from load_bridge_cases), appending to out_path. Returns how many were new."""
    items = (
        ({"case_id": c["case_id"]}, {"history": c["history"], "lesson_topic": c.get("lesson_topic")}, {})
        for c in cases
    )
    return run_cached(
        items, out_path, ("case_id",), prompt_version(), lambda payload: judge_acceptable(llm, **payload),
        skip_errors=(AcceptableError, LLMError),
        usage=getattr(llm, "usage", None),
    )


def load_acceptable(path, version: Optional[str] = None) -> dict[str, list[str]]:
    """{case_id: [acceptable actions]} from build_acceptable output (.jsonl) or a hand-made .json dict.

    A .jsonl file holding several prompt versions needs `version` to pick one.
    """
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    versions = sorted({r["prompt_version"] for r in rows})
    if version is None:
        if len(versions) > 1:
            raise ValueError(f"{path} has prompt versions {versions}; pass version= to pick one")
        version = versions[0] if versions else None
    return {r["case_id"]: r["acceptable"] for r in rows if r["prompt_version"] == version}


def summarize(acceptable: dict[str, list[str]], gold: dict) -> dict:
    """Sanity checks before trusting the sets.

    expert_coverage: share of expert annotations whose action the LLM also
    accepted. Low coverage means the LLM disagrees with experts a lot, so its
    extra actions are suspect too. mean_set_size near 7 means it accepts
    everything, which would make lenient scoring meaningless.
    """
    cases = [c for c in acceptable if c in gold]
    expert = [a in acceptable[c] for c in cases for a in gold[c]["actions"]]
    counts = Counter(a for c in cases for a in acceptable[c])
    return {
        "n_cases": len(cases),
        "mean_set_size": statistics.fmean(len(acceptable[c]) for c in cases) if cases else 0.0,
        "expert_coverage": statistics.fmean(expert) if expert else None,
        "accept_rate": {a: counts[a] / len(cases) for a in ACTIONS} if cases else {},
    }


def export_review_sheet(cases: Sequence[dict], acceptable: dict, out_path, n: int, seed: int = 0) -> None:
    """Blind review sheet: the reviewer sees the conversation only (no LLM or expert
    labels) and types y or n under each action."""
    judged = [c for c in cases if c["case_id"] in acceptable]
    sample = random.Random(seed).sample(judged, min(n, len(judged)))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, ["case_id", "lesson_topic", "conversation"] + ACTIONS)
        writer.writeheader()
        for c in sample:
            writer.writerow({
                "case_id": c["case_id"],
                "lesson_topic": c.get("lesson_topic") or "",
                "conversation": "\n".join(f"{t['role'].capitalize()}: {t['text']}" for t in c["history"]),
            })


YES, NO = {"y", "yes", "1", "true"}, {"n", "no", "0", "false"}


def read_review_sheet(path) -> dict[str, dict[str, bool]]:
    """{case_id: {action: bool}} for fully filled rows; blank rows are skipped."""
    reviewed = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cells = {a: (row.get(a) or "").strip().casefold() for a in ACTIONS}
            if not any(cells.values()):
                continue
            bad = [a for a, v in cells.items() if v not in YES | NO]
            if bad:
                raise ValueError(f"{path}: {row['case_id']} needs y or n for {bad}")
            reviewed[row["case_id"]] = {a: v in YES for a, v in cells.items()}
    return reviewed


def review_agreement(sheet_path, acceptable: dict[str, list[str]]) -> dict:
    """LLM vs human on the reviewed cases: kappa per action, plus overall precision
    (of actions the LLM accepted, share the human accepted) and recall."""
    human = read_review_sheet(sheet_path)
    cases = sorted(c for c in human if c in acceptable)
    if not cases:
        raise ValueError("no reviewed cases match the acceptable sets")
    per_action = {}
    for a in ACTIONS:
        h = [human[c][a] for c in cases]
        m = [a in acceptable[c] for c in cases]
        per_action[a] = {
            # kappa is undefined (nan) if both raters give the same answer for every case
            "kappa": float(cohen_kappa_score(h, m)),
            "agreement": sum(x == y for x, y in zip(h, m)) / len(cases),
        }
    pairs = [(human[c][a], a in acceptable[c]) for c in cases for a in ACTIONS]
    llm_yes = [h for h, m in pairs if m]
    human_yes = [m for h, m in pairs if h]
    return {
        "n_cases": len(cases),
        "precision": statistics.fmean(llm_yes) if llm_yes else None,
        "recall": statistics.fmean(human_yes) if human_yes else None,
        "per_action": per_action,
    }


def run_build(args) -> None:
    """The `build` command: one LLM call per Bridge history not already in the output file."""
    from eval.run_judge import build_llm, load_env_file
    from experiments.run_experiment import DEFAULT_MODELS

    load_env_file()
    model = args.model or DEFAULT_MODELS[args.provider]
    cases = load_bridge_cases(args.bridge)
    if args.limit is not None:
        cases = cases[:args.limit]
    out = Path(args.out or Path("results") / f"acceptable_{Path(args.bridge).stem}_{model.replace('/', '-')}.jsonl")
    version = prompt_version()
    already = _done_cases(out)
    done = sum(1 for c in cases if (c["case_id"], version) in already)
    print(f"{len(cases)} histories, model {args.provider}/{model}, prompt version {version}")
    print(f"output: {out}  ({done} already built with this prompt version, {len(cases) - done} API calls to make)")
    if args.dry_run:
        print("\n--- system prompt ---\n" + system_prompt())
        print("\n--- first user prompt ---\n" + build_user_prompt(cases[0]["history"], lesson_topic=cases[0].get("lesson_topic")))
        print("\ndry run: no API calls made")
        return
    llm = build_llm(args.provider, model, args.effort, args.max_tokens)
    new = build_acceptable(llm, cases, out)
    print(f"built {new} new acceptable-action sets")
    usage = getattr(llm, "usage", None)
    if usage:
        print(f"LLM usage: {usage['calls']} calls, ${usage['cost_usd']:.4f}")
    print(f"next: python -m eval.acceptable summary {out} --bridge {args.bridge}")


def _done_cases(out: Path) -> set[tuple]:
    if not out.exists():
        return set()
    with open(out, encoding="utf-8") as f:
        return {(r["case_id"], r["prompt_version"]) for r in map(json.loads, filter(str.strip, f))}


def main() -> None:
    from eval.load import load_bridge_gold
    from eval.run_judge import PROVIDERS

    parser = argparse.ArgumentParser(description="Acceptable-action sets for lenient strategy scoring")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build the sets with an LLM (costs API credit; try --dry-run first)")
    build.add_argument("--bridge", required=True, help="Bridge split file to build sets for")
    build.add_argument("--limit", type=int, default=None, help="only the first N histories")
    build.add_argument("--provider", choices=PROVIDERS, default="openrouter")
    build.add_argument("--model", default=None, help="model name; omit to use the group default")
    build.add_argument("--effort", choices=["low", "medium", "high"], default="low")
    build.add_argument("--max-tokens", type=int, default=4000)
    build.add_argument("--out", default=None)
    build.add_argument("--dry-run", action="store_true", help="show what would run without calling the API")
    summary = sub.add_parser("summary")
    summary.add_argument("acceptable_file")
    summary.add_argument("--bridge", required=True, help="Bridge split file the sets were built from")
    export = sub.add_parser("export")
    export.add_argument("acceptable_file")
    export.add_argument("--bridge", required=True)
    export.add_argument("--n", type=int, default=30)
    export.add_argument("--seed", type=int, default=0)
    export.add_argument("--out", required=True)
    review = sub.add_parser("review")
    review.add_argument("sheet")
    review.add_argument("acceptable_file")
    args = parser.parse_args()

    if args.command == "build":
        run_build(args)
        return
    acceptable = load_acceptable(args.acceptable_file)
    if args.command == "summary":
        s = summarize(acceptable, load_bridge_gold(args.bridge))
        print(f"cases={s['n_cases']} mean set size={s['mean_set_size']:.2f} "
              f"expert coverage={s['expert_coverage']:.3f}")
        for action, rate in s["accept_rate"].items():
            print(f"  {action:<10} accepted in {rate:.0%} of cases")
    elif args.command == "export":
        export_review_sheet(load_bridge_cases(args.bridge), acceptable, args.out, args.n, args.seed)
        print(f"wrote {args.out}: type y or n under each action, without looking at the LLM's sets")
    else:
        r = review_agreement(args.sheet, acceptable)
        print(f"cases={r['n_cases']} LLM precision={r['precision']:.3f} recall={r['recall']:.3f}")
        for action, stats in r["per_action"].items():
            print(f"  {action:<10} kappa={stats['kappa']:.3f} agreement={stats['agreement']:.3f}")


if __name__ == "__main__":
    main()
