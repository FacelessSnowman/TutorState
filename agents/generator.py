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
# Signs of the model's own planning notes: a tutor talks *to* the student ("you"),
# never about "the student" or about what it "should" do.
_PLANNING_NOTES = re.compile(r"\b(the student|the tutor|the learner)\b|\bI should\b", re.IGNORECASE)
# The tutor claiming it cannot see the problem/answer/work (BRIDGE transcripts omit the
# whiteboard). Rewritten by retrying; a real tutor in the session could see it.
_CANT_SEE = re.compile(
    # "I can't see your work", "I don't know the problem yet"
    r"\b(can.?t|cannot|can not|don.?t|do not|isn.?t|is not|not)\s+(yet\s+)?"
    r"(see|seen|shown|showing|visible|have|know)\b[^.?!]{0,40}"
    r"\b(problem|question|answer|work|whiteboard|picture|screen|number you wrote)\b"
    # "The problem isn't shown in our chat", "the question itself isn't here"
    r"|\b(problem|question)\b[^.?!]{0,20}\b(isn.?t|is not|not)\s+(yet\s+)?"
    r"(shown|showing|visible|in (our|the) chat|here)\b",
    re.IGNORECASE,
)


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


def remove_planning_notes(text: str) -> Optional[str]:
    """Drop paragraphs where the model talks about the student or its plan instead of to the student.

    Returns the remaining message, or None if nothing usable is left (the
    whole reply was notes), so the caller can retry.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    kept = [p for p in paragraphs if not _PLANNING_NOTES.search(p)]
    return "\n\n".join(kept) or None


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
    max_attempts: int = 2,
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
    rejected = []
    fallback = None
    for _ in range(max_attempts):
        raw = llm.complete(system_prompt(action), user)
        response = remove_planning_notes(clean_response(raw))
        if response and not _CANT_SEE.search(response):
            return response
        if response:
            fallback = response  # usable but says it can't see the problem; retry for a better one
        rejected.append(raw)
    if fallback:
        return fallback  # a valid message is better than failing the case
    raise GeneratorError(
        f"no usable message after {max_attempts} attempts (empty or only planning notes): {rejected[-1][:200]!r}"
    )
