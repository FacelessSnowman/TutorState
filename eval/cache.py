"""Append-only JSONL cache for paid LLM calls (judge, acceptable-action sets).

Each output line holds the item's ids, the prompt version and the LLM result.
Items already in the file with the same ids and prompt version are skipped,
so re-running a script never pays twice, and editing a prompt (new version)
re-runs everything for that version only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Sequence


def done_keys(out_path: Path, key_fields: Sequence[str]) -> set[tuple]:
    if not out_path.exists():
        return set()
    with open(out_path, encoding="utf-8") as f:
        return {
            tuple(row[k] for k in key_fields) + (row["prompt_version"],)
            for row in map(json.loads, filter(str.strip, f))
        }


def run_cached(
    items: Iterable[tuple[dict, dict, dict]],
    out_path,
    key_fields: Sequence[str],
    version: str,
    compute: Callable[[dict], dict],
) -> int:
    """items: (ids, payload, extra) triples. compute(payload) returns the result
    fields for the line; extra is copied into the line but never sent to the LLM.
    Returns how many new lines were written."""
    out_path = Path(out_path)
    done = done_keys(out_path, key_fields)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for ids, payload, extra in items:
            if tuple(ids[k] for k in key_fields) + (version,) in done:
                continue
            row = {**ids, "prompt_version": version, **compute(payload), **extra}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()  # each line is paid for; keep it even if a later call crashes
            new += 1
    return new
