"""Prompt loading and input formatting shared by the planner and generator.

Every optional input (analysis, memory, action, ...) is rendered as its own
section and simply left out when it is None. This is what makes the ablation
fair: each condition uses the same prompt, minus the sections for components
that are switched off.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

from schemas import ACTIONS, Turn

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def action_definitions() -> dict[str, str]:
    """{"PROBE": "Ask the student ...", ...} parsed from prompts/action_definitions.txt."""
    definitions = {}
    for line in load_prompt("action_definitions.txt").splitlines():
        if line.strip():
            name, text = line.split(":", 1)
            definitions[name.strip()] = text.strip()
    missing = set(ACTIONS) - set(definitions)
    if missing:
        raise ValueError(f"action_definitions.txt is missing {sorted(missing)}")
    return definitions


def format_action_definitions() -> str:
    return "\n".join(f"- {name}: {text}" for name, text in action_definitions().items())


def format_history(history: Sequence[Turn]) -> str:
    if not history:
        raise ValueError("history is empty")
    if history[-1]["role"] != "student":
        raise ValueError("history must end with a student turn")
    return "\n".join(f"{t['role'].capitalize()}: {t['text']}" for t in history)


def _format_value(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def format_fields(fields: dict) -> str:
    """Render an analysis/memory dict as bullets. Generic so schema renames don't break it."""
    return "\n".join(f"- {key.replace('_', ' ')}: {_format_value(value)}" for key, value in fields.items())


def build_user_prompt(
    history: Sequence[Turn],
    lesson_topic: Optional[str] = None,
    problem: Optional[str] = None,
    reference_solution: Optional[str] = None,
    analysis: Optional[dict] = None,
    memory: Optional[dict] = None,
    extra_sections: Sequence[tuple[str, str]] = (),
) -> str:
    context = []
    if lesson_topic:
        context.append(f"Lesson topic: {lesson_topic}")
    if problem:
        context.append(f"Problem: {problem}")
    if reference_solution:
        context.append(f"Reference solution (for your understanding only, never reveal it directly): {reference_solution}")

    sections = []
    if context:
        sections.append(("Context", "\n".join(context)))
    sections.append((
        "Conversation so far (the last line is the student's latest message)",
        format_history(history),
    ))
    if analysis is not None:
        sections.append(("Diagnosis of the student's latest message", format_fields(analysis)))
    if memory is not None:
        sections.append(("Student state across the session", format_fields(memory)))
    sections.extend(extra_sections)

    return "\n\n".join(f"## {title}\n{body}" for title, body in sections)
