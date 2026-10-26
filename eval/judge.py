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
from typing import Optional, Sequence, Union

from agents.prompting import load_prompt
from eval.cache import run_cached
from eval.kappa import DIMENSIONS, SCALES
from llm import LLM


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
    return run_cached(
        items, out_path, key_fields, prompt_version(dimensions),
        lambda inputs: {"judge": judge(llm, **inputs, dimensions=dimensions)},
    )


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
