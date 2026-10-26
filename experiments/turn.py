import time
from typing import Optional

from agents.generator import generate
from agents.planner import plan

from experiments.adapters import (
    run_analyser,
    run_memory,
)
from experiments.conditions import CONDITIONS


def build_prior_memory(
    *,
    llm,
    case: dict,
    history: list[dict],
    condition_name: str,
    previous_memory: Optional[dict] = None,
    memory_profile: Optional[str] = None,
) -> tuple[Optional[dict], list[dict]]:
    """Fold every student turn *before* the last one into memory (analyser -> memory).

    This is the agreed harness loop: earlier student turns only update memory,
    and run_turn then handles the final student turn with all components. Each
    earlier turn is analysed with only the history up to that turn, so no later
    information leaks in. Conditions without memory skip this (nothing persists).

    Returns (memory, warm-up steps for the trajectory log).
    """
    condition = CONDITIONS[condition_name]
    if not condition.use_memory:
        return previous_memory, []

    memory = previous_memory
    steps = []
    student_positions = [i for i, turn in enumerate(history) if turn["role"] == "student"]

    for position in student_positions[:-1]:
        prefix = history[: position + 1]
        analysis = run_analyser(
            llm=llm,
            history=prefix,
            lesson_topic=case.get("lesson_topic"),
            problem=case.get("problem"),
        )
        memory = run_memory(
            history=prefix,
            analysis=analysis,
            previous_memory=memory,
            memory_profile=memory_profile,
        )
        steps.append(
            {
                "step": f"warmup_{len(steps)}",
                "history_before": prefix,
                "analysis": analysis,
                "memory": memory,
            }
        )

    return memory, steps


def run_turn(
    *,
    llm,
    case: dict,
    history: list[dict],
    condition_name: str,
    previous_memory: Optional[dict] = None,
    memory_profile: Optional[str] = None,
) -> dict:

    condition = CONDITIONS[condition_name]

    analysis = None
    memory = None
    planner_output = None
    timings = {}

    # Analyser -> Implemented
    if condition.use_analyser:
        start = time.perf_counter()

        analysis = run_analyser(
            llm=llm,
            history=history,
            lesson_topic=case.get("lesson_topic"),
            problem=case.get("problem"),
        )

        timings["analyser_ms"] = (
            time.perf_counter() - start
        ) * 1000

    # Memory -> Implemented
    if condition.use_memory:
        start = time.perf_counter()

        memory = run_memory(
            history=history,
            analysis=analysis,
            previous_memory=previous_memory,
            memory_profile=memory_profile,
        )

        timings["memory_ms"] = (
            time.perf_counter() - start
        ) * 1000

    # Planner -> Implemented
    if condition.use_planner:
        start = time.perf_counter()

        planner_output = plan(
            llm=llm,
            history=history,
            lesson_topic=case.get("lesson_topic"),
            problem=case.get("problem"),
            analysis=analysis,
            memory=memory,
        )

        timings["planner_ms"] = (
            time.perf_counter() - start
        ) * 1000

    # Generator -> Implemented
    start = time.perf_counter()

    response = generate(
        llm=llm,
        history=history,
        lesson_topic=case.get("lesson_topic"),
        problem=case.get("problem"),
        reference_solution=case.get("reference_solution"),
        analysis=analysis,
        memory=memory,
        action=(
            planner_output["action"]
            if planner_output
            else None
        ),
        planner_reason=(
            planner_output["reason"]
            if planner_output
            else None
        ),
    )

    timings["generator_ms"] = (
        time.perf_counter() - start
    ) * 1000

    return {
        "analysis": analysis,
        "memory": memory,
        "planner": planner_output,
        "response": response,
        "timings": timings,
    }