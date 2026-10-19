"""Response generator (Track C): writes the tutor's next message.

    generate(llm, history, lesson_topic, action, analysis, memory) -> str

One template serves every ablation condition. Inputs that a condition does not
have are passed as None and their prompt sections are left out:
    baseline          generate(llm, history, lesson_topic)
    + analyser        generate(..., analysis=a)
    + memory          generate(..., analysis=a, memory=m)
    full TutorState   generate(..., analysis=a, memory=m, action=p["action"], planner_reason=p["reason"])
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from agents.prompting import action_definitions, build_user_prompt, load_prompt
from llm import LLM
from schemas import ACTIONS, Turn

_SPEAKER_LABEL = re.compile(r"^\s*\**\s*(tutor|teacher|assistant)\s*\**\s*:\s*\**\s*", re.IGNORECASE)
_WRAPPING_QUOTES = ('"', "'", "“", "”")


class GeneratorError(RuntimeError):
    pass


def system_prompt(action: Optional[str]) -> str:
    action_rules = ""
    if action is not None:
        action_rules = (
            f"- Your teaching move for this message is {action}: {action_definitions()[action]}\n"
            f"- Make exactly this move and no other. Do not add a second move, for example a "
            f"HINT must not go on to explain the whole method.\n"
        )
    return load_prompt("generator_system.txt").format(action_rules=action_rules)


def clean_response(text: str) -> str:
    """Strip artefacts LLMs add around chat messages (speaker labels, wrapping quotes)."""
    text = _SPEAKER_LABEL.sub("", text.strip()).strip()
    if len(text) >= 2 and text[0] in _WRAPPING_QUOTES and text[-1] in _WRAPPING_QUOTES:
        text = text[1:-1].strip()
    return text


def generate(
    llm: LLM,
    history: Sequence[Turn],
    lesson_topic: Optional[str] = None,
    action: Optional[str] = None,
    analysis: Optional[dict] = None,
    memory: Optional[dict] = None,
    planner_reason: Optional[str] = None,
    problem: Optional[str] = None,
    reference_solution: Optional[str] = None,
) -> str:
    if action is not None and action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")

    extra = []
    if action is not None:
        move = action if not planner_reason else f"{action}\nWhy: {planner_reason}"
        extra.append(("Teaching move to make", move))

    user = build_user_prompt(
        history,
        lesson_topic=lesson_topic,
        problem=problem,
        reference_solution=reference_solution,
        analysis=analysis,
        memory=memory,
        extra_sections=extra,
    )
    response = clean_response(llm.complete(system_prompt(action), user))
    if not response:
        raise GeneratorError("model returned an empty message")
    return response
