"""Thin adapters between the runner and the agents, so agent signatures can change
without touching the runner loop."""

from typing import Optional

from agents.analyser import analyse
from agents.memory import update_memory


def run_analyser(
    *,
    llm,
    history,
    lesson_topic=None,
    problem=None,
):
    return analyse(
        llm,
        history,
        lesson_topic=lesson_topic,
        problem=problem,
    )


def run_memory(
    *,
    history,
    analysis,
    previous_memory=None,
    memory_profile=None,
):
    # memory_profile is reserved for the scripted memory-consistency profiles;
    # the memory manager itself is plain rules and needs no LLM.
    return update_memory(
        previous_memory,
        analysis,
        history,
    )


def run_student_simulator(
    *,
    llm,
    history,
    case,
    step,
):
    raise RuntimeError(
        "Student simulator has not been integrated yet."
    )
