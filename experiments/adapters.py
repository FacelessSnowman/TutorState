from typing import Optional


def run_analyser(
    *,
    llm,
    history,
    lesson_topic=None,
    problem=None,
):
    raise RuntimeError(
        "Student analyser has not been integrated yet."
    )


def run_memory(
    *,
    history,
    analysis,
    previous_memory=None,
    memory_profile=None,
):
    raise RuntimeError(
        "Student memory has not been integrated yet."
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