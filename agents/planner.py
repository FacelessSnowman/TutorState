"""Pedagogical planner (Track C): chooses one teaching action for the next tutor turn.

    plan(llm, history, lesson_topic, analysis, memory) -> {"reason": str, "action": str}

The output is constrained to the action enum by a JSON schema, then validated
again here. A malformed output is retried; if it still fails, PlannerError is
raised. There is deliberately no silent default action, because a default
would quietly bias the strategy confusion matrix.
"""

from __future__ import annotations

import json
from typing import Optional, Sequence

from agents.prompting import build_user_prompt, format_action_definitions, load_prompt
from llm import LLM
from schemas import ACTIONS, PlannerOutput, Turn

PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "action": {"type": "string", "enum": ACTIONS},
    },
    "required": ["reason", "action"],
    "additionalProperties": False,
}


class PlannerError(RuntimeError):
    def __init__(self, message: str, raw_outputs: list[str]):
        super().__init__(message)
        self.raw_outputs = raw_outputs


def system_prompt() -> str:
    return load_prompt("planner_system.txt").format(action_definitions=format_action_definitions())


def parse_planner_output(raw: str) -> PlannerOutput:
    """Validate the model's JSON. Raises ValueError with the reason it is invalid."""
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("output is not a JSON object")
    action = str(data.get("action", "")).strip().upper()
    if action not in ACTIONS:
        raise ValueError(f"action {data.get('action')!r} is not one of {ACTIONS}")
    reason = str(data.get("reason", "")).strip()
    if not reason:
        raise ValueError("reason is empty")
    return {"reason": reason, "action": action}


def plan(
    llm: LLM,
    history: Sequence[Turn],
    lesson_topic: Optional[str] = None,
    analysis: Optional[dict] = None,
    memory: Optional[dict] = None,
    problem: Optional[str] = None,
    max_attempts: int = 2,
) -> PlannerOutput:
    """Choose the next action.

    analysis/memory may be None to run the "planner on raw history only" variant.
    """
    user = build_user_prompt(
        history, lesson_topic=lesson_topic, problem=problem, analysis=analysis, memory=memory
    )
    raw_outputs = []
    for _ in range(max_attempts):
        raw = llm.complete(system_prompt(), user, json_schema=PLANNER_SCHEMA)
        raw_outputs.append(raw)
        try:
            return parse_planner_output(raw)
        except (ValueError, json.JSONDecodeError):
            continue
    raise PlannerError(f"no valid action after {max_attempts} attempts", raw_outputs)
