"""LLM judge (Track D): rates one tutor response on the MRBench dimensions.

    judge(llm, conversation_history, ground_truth_solution, candidate_response)
        -> {dimension: {"reasoning": str, "label": str}}

All dimensions are rated in one call to keep API cost down (8x fewer calls
than one call per dimension). The judge only ever sees the conversation, the
reference solution, the candidate response and, for Bridge, the expert
response as context - never human ratings or which model wrote the response.

run_judge() appends one JSON line per response and skips responses already
judged with the same prompt version, so re-running never pays twice.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence, Union

from agents.prompting import load_prompt
from eval.kappa import DIMENSIONS, SCALES
from llm import LLM, BudgetExceeded, LLMError


class JudgeError(RuntimeError):
    def __init__(self, message: str, raw_outputs: list[str]):
        super().__init__(message)
        self.raw_outputs = raw_outputs


def judge_schema(dimensions: Sequence[str] = DIMENSIONS) -> dict:
    return {
        "type": "object",
        "properties": {
            dim: {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "label": {"type": "string", "enum": list(SCALES[dim])},
                },
                "required": ["reasoning", "label"],
                "additionalProperties": False,
            }
            for dim in dimensions
        },
        "required": list(dimensions),
        "additionalProperties": False,
    }


def system_prompt() -> str:
    return load_prompt("judge_system.txt")


def prompt_version(dimensions: Sequence[str] = DIMENSIONS) -> str:
    """Short hash of the prompt and dimensions; editing the prompt invalidates the cache."""
    text = system_prompt() + "|" + ",".join(dimensions)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _format_history(history: Union[str, Sequence[dict]]) -> str:
    if isinstance(history, str):
        return history.strip()
    return "\n".join(f"{t['role'].capitalize()}: {t['text']}" for t in history)


def build_user_prompt(
    conversation_history: Union[str, Sequence[dict]],
    ground_truth_solution: Optional[str],
    candidate_response: str,
    expert_response: Optional[str] = None,
    dimensions: Sequence[str] = DIMENSIONS,
) -> str:
    sections = [
        "## Conversation\n" + _format_history(conversation_history),
        "## Reference solution\n" + (ground_truth_solution.strip() if ground_truth_solution else "Not available."),
    ]
    if expert_response:
        sections.append("## Expert response (context only)\n" + expert_response.strip())
    sections.append("## Candidate tutor response\n" + candidate_response.strip())
    sections.append("## Dimensions to rate\n" + ", ".join(dimensions))
    return "\n\n".join(sections)


def parse_judge_output(raw: str, dimensions: Sequence[str] = DIMENSIONS) -> dict:
    """Validate the model's JSON. Raises ValueError with the reason it is invalid."""
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("output is not a JSON object")
    parsed = {}
    for dim in dimensions:
        entry = data.get(dim)
        if not isinstance(entry, dict):
            raise ValueError(f"{dim} is missing")
        allowed = {label.casefold(): label for label in SCALES[dim]}
        label = allowed.get(str(entry.get("label", "")).strip().casefold())
        if label is None:
            raise ValueError(f"{dim} label {entry.get('label')!r} is not one of {list(SCALES[dim])}")
        parsed[dim] = {"reasoning": str(entry.get("reasoning", "")).strip(), "label": label}
    return parsed


def judge(
    llm: LLM,
    conversation_history: Union[str, Sequence[dict]],
    ground_truth_solution: Optional[str],
    candidate_response: str,
    expert_response: Optional[str] = None,
    dimensions: Sequence[str] = DIMENSIONS,
    max_attempts: int = 2,
) -> dict:
    user = build_user_prompt(
        conversation_history, ground_truth_solution, candidate_response, expert_response, dimensions
    )
    raw_outputs = []
    for _ in range(max_attempts):
        raw = llm.complete(system_prompt(), user, json_schema=judge_schema(dimensions))
        raw_outputs.append(raw)
        try:
            return parse_judge_output(raw, dimensions)
        except (ValueError, json.JSONDecodeError):
            continue
    raise JudgeError(f"no valid ratings after {max_attempts} attempts", raw_outputs)


MRBENCH_KEY = ("conversation_id", "response_source")
RESULTS_KEY = ("case_id", "condition", "seed")


def _judge_to_file(llm: LLM, items, out_path, key_fields: Sequence[str], dimensions: Sequence[str]) -> int:
    """items: (ids, inputs, extra) triples. ids identify the response, inputs go to
    judge(), extra is copied into the output line but never into the prompt."""
    out_path = Path(out_path)
    version = prompt_version(dimensions)
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for row in map(json.loads, filter(str.strip, f)):
                done.add(tuple(row[k] for k in key_fields) + (row["prompt_version"],))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    new = failed = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for ids, inputs, extra in items:
            if tuple(ids[k] for k in key_fields) + (version,) in done:
                continue
            try:
                ratings = judge(llm, **inputs, dimensions=dimensions)
            except BudgetExceeded:
                raise  # out of money: stop; judged lines are already saved
            except (JudgeError, LLMError) as error:
                # Not written, so the next run retries it.
                failed += 1
                print(f"  judge failed on {ids}: {type(error).__name__}: {str(error)[:150]}")
                continue
            row = {**ids, "prompt_version": version, "judge": ratings, **extra}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()  # each line is paid for; keep it even if a later call crashes
            new += 1
            if new % 10 == 0:
                print(f"  judged {new} so far (${getattr(llm, 'usage', {}).get('cost_usd', 0.0):.4f})", flush=True)
    if failed:
        print(f"  {failed} response(s) could not be judged; rerun to retry them")
    return new


def run_judge(llm: LLM, examples: Sequence[dict], out_path, dimensions: Sequence[str] = DIMENSIONS) -> int:
    """Judge MRBench's own candidate responses (from load_mrbench_examples), to validate the judge.

    Returns how many new responses were judged. human_ratings is copied into
    the output line for scoring only; it is never put in the prompt.
    """
    items = (
        (
            {k: ex[k] for k in MRBENCH_KEY},
            {
                "conversation_history": ex["conversation_history"],
                "ground_truth_solution": ex["ground_truth_solution"],
                "candidate_response": ex["candidate_response"],
            },
            {"human_ratings": ex.get("human_ratings")},
        )
        for ex in examples
    )
    return _judge_to_file(llm, items, out_path, MRBENCH_KEY, dimensions)


def run_judge_results(
    llm: LLM, results: Sequence[dict], contexts: dict, out_path, dimensions: Sequence[str] = DIMENSIONS
) -> int:
    """Judge TutorState's responses from the results file.

    contexts comes from load_bridge_contexts or load_mrbench_contexts. The judge
    never sees the condition or seed, so it cannot favour a condition.
    """
    missing = sorted({r["case_id"] for r in results} - set(contexts))
    if missing:
        raise ValueError(f"{len(missing)} case_ids have no context, e.g. {missing[:3]}")
    items = (
        ({k: r[k] for k in RESULTS_KEY}, {**contexts[r["case_id"]], "candidate_response": r["response"]}, {})
        for r in results
        if r.get("response")
    )
    return _judge_to_file(llm, items, out_path, RESULTS_KEY, dimensions)


def main() -> None:
    """Command-line entry point.

    1. Validate the judge against MRBench's human ratings (do this first):
         python -m eval.judge mrbench --split train --n 100 --out results/judge_mrbench_train.jsonl
         python -m eval.kappa results/judge_mrbench_train.jsonl
    2. Judge TutorState's responses from a runner results file:
         python -m eval.judge results --results results/run.jsonl \\
             --bridge data/Bridge/bridge_data_size_441/441_test.json --out results/judge_run.jsonl
         python -m eval.report --results results/run.jsonl --bridge ... --judge results/judge_run.jsonl

    Uses the group LLM (llm.get_llm); --model swaps the judge model, e.g. a
    different family from the agents to avoid self-preference. Re-running the
    same command skips responses already judged with the current prompt.
    """
    import argparse
    import random

    from eval.load import (
        load_bridge_contexts,
        load_mrbench_contexts,
        load_mrbench_examples,
        load_results,
    )
    from llm import get_llm

    parser = argparse.ArgumentParser(description="LLM judge on the MRBench dimensions")
    parser.add_argument("--model", help="OpenRouter model slug; default is the group model in llm.py")
    parser.add_argument("--effort", help="reasoning effort; default is the group setting in llm.py")
    sub = parser.add_subparsers(dest="command", required=True)

    mrbench = sub.add_parser("mrbench", help="judge MRBench's own responses, to compare with human ratings")
    mrbench.add_argument("--split", default="train", choices=["train", "validation", "test"])
    mrbench.add_argument("--n", type=int, default=None, help="random sample size (default: all)")
    mrbench.add_argument("--seed", type=int, default=0, help="sampling seed, so the sample is reproducible")
    mrbench.add_argument("--out", required=True)

    results = sub.add_parser("results", help="judge TutorState responses from a runner results file")
    results.add_argument("--results", required=True)
    source = results.add_mutually_exclusive_group(required=True)
    source.add_argument("--bridge", help="BRIDGE split file the results were run on")
    source.add_argument("--mrbench-split", choices=["train", "validation", "test"])
    results.add_argument("--out", required=True)

    args = parser.parse_args()
    overrides = {k: v for k, v in (("model", args.model), ("effort", args.effort)) if v}
    llm = get_llm(**overrides)
    print(f"judge model: {llm.model} (effort={llm.effort}), prompt version {prompt_version()}")

    if args.command == "mrbench":
        examples = load_mrbench_examples(args.split)
        if args.n is not None and args.n < len(examples):
            examples = random.Random(args.seed).sample(examples, args.n)
        new = run_judge(llm, examples, args.out)
    else:
        rows = load_results(args.results)
        contexts = load_bridge_contexts(args.bridge) if args.bridge else load_mrbench_contexts(args.mrbench_split)
        new = run_judge_results(llm, rows, contexts, args.out)

    usage = llm.usage
    print(f"judged {new} new response(s) -> {args.out}")
    print(f"LLM usage: {usage['calls']} calls, ${usage['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
