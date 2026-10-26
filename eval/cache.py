"""Append-only JSONL cache for paid LLM calls (judge, acceptable-action sets).

Each output line holds the item's ids, the prompt version and the LLM result.
Items already in the file with the same ids and prompt version are skipped,
so re-running a script never pays twice, and editing a prompt (new version)
re-runs everything for that version only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from llm import BudgetExceeded


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
    skip_errors: tuple = (),
    usage: Optional[dict] = None,
    progress_every: int = 10,
) -> int:
    """items: (ids, payload, extra) triples. compute(payload) returns the result
    fields for the line; extra is copied into the line but never sent to the LLM.
    Returns how many new lines were written.

    skip_errors: exceptions that mean "this one item failed" (e.g. invalid
    output after retries). The item is not written, so the next run retries
    it, and the loop moves on. BudgetExceeded always stops the run.
    usage: the LLM's running usage dict, to print cost with the progress line.
    """
    out_path = Path(out_path)
    done = done_keys(out_path, key_fields)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new = failed = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for ids, payload, extra in items:
            if tuple(ids[k] for k in key_fields) + (version,) in done:
                continue
            try:
                result = compute(payload)
            except BudgetExceeded:
                raise  # out of money: stop; finished lines are already saved
            except skip_errors as error:
                failed += 1
                print(f"  failed on {ids}: {type(error).__name__}: {str(error)[:150]}", flush=True)
                continue
            row = {**ids, "prompt_version": version, **result, **extra}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()  # each line is paid for; keep it even if a later call crashes
            new += 1
            if progress_every and new % progress_every == 0:
                cost = f" (${usage.get('cost_usd', 0.0):.4f})" if usage else ""
                print(f"  {new} done so far{cost}", flush=True)
    if failed:
        print(f"  {failed} item(s) failed and were not saved; rerun the same command to retry them")
    return new
