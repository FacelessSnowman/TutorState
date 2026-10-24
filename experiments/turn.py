import time
from typing import Optional

from agents.generator import generate
from agents.planner import plan

from experiments.adapters import (
    run_analyser,
    run_memory,
)
from experiments.conditions import CONDITIONS


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

    # Analyser -> this isnt implemented yet
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

    # Memory -> this isnt implemented yet
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

    # Planner -> this isnt implemented yet
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