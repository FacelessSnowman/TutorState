"""Student analyser (Track B): diagnoses the student's latest message.

    analyse(llm, history, lesson_topic, problem) -> Analysis (schemas.py, proposal Table 2)

The model first writes a short `reasoning` (what was asked, what the correct
answer is, how the student's answer compares) and only then picks the labels,
because labelling after working it out is more accurate. The reasoning is
dropped from the returned Analysis by default: it often states the correct
answer, and the analysis is shown to the generator, so passing it on could
make the tutor reveal the answer.

`error_type` is constrained to the BRIDGE e_clean labels so Track D can score
it directly against the expert labels. As with the planner, an invalid output
is retried and then raises AnalyserError; there is no silent default label,
because a default would quietly bias the diagnosis confusion matrix.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Optional, Sequence, get_args

from agents.prompting import build_user_prompt, load_prompt
from llm import LLM
from schemas import Analysis, ErrorType, Turn

ERROR_TYPES = list(get_args(ErrorType))
LEVELS = ["low", "medium", "high"]

ANALYSER_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "correct": {"type": ["boolean", "null"]},
        "concept": {"type": ["string", "null"]},
        "error_type": {"type": "string", "enum": ERROR_TYPES},
        "misconception": {"type": ["string", "null"]},
        "confidence": {"type": "string", "enum": LEVELS},
    },
    "required": ["reasoning", "correct", "concept", "error_type", "misconception", "confidence"],
    "additionalProperties": False,
}


class AnalyserError(RuntimeError):
    def __init__(self, message: str, raw_outputs: list[str]):
        super().__init__(message)
        self.raw_outputs = raw_outputs


@lru_cache(maxsize=None)
def error_type_definitions() -> dict[str, str]:
    """{"guess": "The answer has no sensible link ...", ...} from prompts/error_type_definitions.txt."""
    definitions = {}
    for line in load_prompt("error_type_definitions.txt").splitlines():
        if line.strip():
            name, text = line.split(":", 1)
            definitions[name.strip()] = text.strip()
    missing = set(ERROR_TYPES) - set(definitions)
    if missing:
        raise ValueError(f"error_type_definitions.txt is missing {sorted(missing)}")
    return definitions


def system_prompt() -> str:
    definitions = "\n".join(f"- {name}: {text}" for name, text in error_type_definitions().items())
    return load_prompt("analyser_system.txt").format(error_type_definitions=definitions)


def _optional_text(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in ("", "none", "null", "n/a") else text


def parse_analyser_output(raw: str, include_reasoning: bool = False) -> Analysis:
    """Validate and tidy the model's JSON. Raises ValueError with the reason it is invalid."""
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("output is not a JSON object")

    error_type = str(data.get("error_type", "")).strip().lower()
    if error_type not in ERROR_TYPES:
        raise ValueError(f"error_type {data.get('error_type')!r} is not one of {ERROR_TYPES}")

    confidence = str(data.get("confidence", "")).strip().lower()
    if confidence not in LEVELS:
        raise ValueError(f"confidence {data.get('confidence')!r} is not one of {LEVELS}")

    correct = data.get("correct")
    if correct not in (True, False, None):
        raise ValueError(f"correct {correct!r} is not true, false or null")
    # Keep the two fields consistent: no_error means the answer was right.
    if error_type == "no_error":
        correct = True

    misconception = _optional_text(data.get("misconception"))
    if error_type == "no_error":
        misconception = None

    analysis: dict = {
        "correct": correct,
        "concept": _optional_text(data.get("concept")),
        "error_type": error_type,
        "misconception": misconception,
        "confidence": confidence,
    }
    if include_reasoning:
        analysis["reasoning"] = str(data.get("reasoning", "")).strip()
    return analysis


def analyse(
    llm: LLM,
    history: Sequence[Turn],
    lesson_topic: Optional[str] = None,
    problem: Optional[str] = None,
    include_reasoning: bool = False,
    max_attempts: int = 2,
) -> Analysis:
    """Diagnose the last (student) message of `history`.

    Only the conversation, topic and (if available) problem are shown to the
    model, never expert labels or reference responses.
    """
    user = build_user_prompt(history, lesson_topic=lesson_topic, problem=problem)
    raw_outputs = []
    for _ in range(max_attempts):
        raw = llm.complete(system_prompt(), user, json_schema=ANALYSER_SCHEMA)
        raw_outputs.append(raw)
        try:
            return parse_analyser_output(raw, include_reasoning=include_reasoning)
        except (ValueError, json.JSONDecodeError):
            continue
    raise AnalyserError(f"no valid analysis after {max_attempts} attempts", raw_outputs)
