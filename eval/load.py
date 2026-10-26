"""Loaders for Track D: the pipeline results file and the gold labels.

Results file (written by Track E's runner) - JSONL, one line per
case / condition / seed:

    {"dataset": "bridge142", "split": "test", "case_id": "H_...",
     "condition": "full", "seed": 0,
     "analysis": {...Analysis...} | null,
     "memory": {...Memory...} | null,
     "planner": {"reason": "...", "action": "HINT"} | null,
     "response": "..."}

See eval/results_example.jsonl for complete lines. analysis, memory and
planner use the schemas.py field names; Track D reads analysis.error_type,
planner.action and response. Components that are switched off in a
condition are null. Gold labels are
never written into this file; they are joined here by case_id, so only
evaluation code ever reads them.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_KEYS = ("case_id", "condition", "seed")


def load_results(path) -> list[dict]:
    """One row per case/condition/seed.

    The runner writes a row with an "error" field when a case fails and
    retries it on the next run, so a key can appear as one or more failures
    followed by a success. The latest row wins; a key that never succeeded
    keeps its error row so failure_rate() can count it. Two successful rows
    for one key is still an error.
    """
    rows: dict = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [k for k in REQUIRED_KEYS if k not in row]
            if missing:
                raise ValueError(f"{path}:{line_no} is missing {missing}")
            key = (row["case_id"], row["condition"], row["seed"])
            previous = rows.get(key)
            if previous is not None and not previous.get("error") and not row.get("error"):
                raise ValueError(f"{path}:{line_no} duplicates case/condition/seed {key}")
            if previous is None or previous.get("error"):
                rows[key] = row
    return list(rows.values())


def failure_rate(results) -> dict[str, dict]:
    """{condition: {"n", "failed", "rate", "by_type"}} over rows still failed after retries.

    Failed rows have no analysis/planner/response, so the scorers skip them;
    report this alongside every score, since failures may not be random
    (e.g. the planner failing more often on unclear student messages).
    """
    report: dict = {}
    for row in results:
        stats = report.setdefault(row["condition"], {"n": 0, "failed": 0, "by_type": {}})
        stats["n"] += 1
        if row.get("error"):
            stats["failed"] += 1
            kind = row["error"].get("type", "unknown")
            stats["by_type"][kind] = stats["by_type"].get(kind, 0) + 1
    for stats in report.values():
        stats["rate"] = stats["failed"] / stats["n"]
    return report


def load_bridge_gold(path) -> dict[str, dict]:
    """{history_id: {"e": [...], "actions": [...], "expert_responses": [...]}}.

    One list entry per expert annotation, in file order and with duplicates
    kept, so two experts agreeing on a label counts twice in fraction_matched.
    """
    gold = {}
    for row in json.loads(Path(path).read_text(encoding="utf-8")):
        annotations = row["annotations"]
        gold[row["history_id"]] = {
            "e": [a["e_clean"] for a in annotations],
            "actions": [a["z_what_clean"] for a in annotations],
            "expert_responses": [" ".join(t["text"] for t in a["c_r_"]) for a in annotations],
        }
    return gold


def load_bridge_contexts(path) -> dict[str, dict]:
    """What the judge sees for a Bridge case. Bridge has no reference solution,
    so the expert responses are given as context instead."""
    contexts = {}
    for row in json.loads(Path(path).read_text(encoding="utf-8")):
        experts = [" ".join(t["text"] for t in a["c_r_"]) for a in row["annotations"]]
        contexts[row["history_id"]] = {
            "conversation_history": row["c_h"],
            "ground_truth_solution": None,
            "expert_response": "\n".join(f"Expert {i}: {text}" for i, text in enumerate(experts, 1)),
        }
    return contexts


def load_mrbench_contexts(split: str) -> dict[str, dict]:
    """What the judge sees for an MRBench case: history and reference solution, keyed by conversation_id."""
    return {
        ex["conversation_id"]: {
            "conversation_history": ex["conversation_history"],
            "ground_truth_solution": ex["ground_truth_solution"],
            "expert_response": None,
        }
        for ex in _mrbench_utils().get_tutor_examples(_mrbench_records(split))
    }


def _mrbench_records(split: str) -> list[dict]:
    return json.loads((ROOT / "data" / "MrBench" / f"{split}.json").read_text(encoding="utf-8"))


def _mrbench_utils():
    spec = importlib.util.spec_from_file_location(
        "mrbench_dataset_utils", ROOT / "data" / "MrBench" / "dataset_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_mrbench_examples(split: str, include_labels: bool = True) -> list[dict]:
    """Flat judge examples (one per candidate response) via Track A's dataset_utils."""
    return _mrbench_utils().get_judge_examples(_mrbench_records(split), include_labels=include_labels, split=split)
