import copy

from experiments.turn import build_prior_memory, run_turn


def run_teacher_forced_case(
    *,
    llm,
    case,
    condition_name,
    memory_profile=None,
):
    """Replay the dataset's history: earlier student turns build memory, the
    final student turn goes through every component switched on."""
    history = copy.deepcopy(case["history"])

    memory, trajectory = build_prior_memory(
        llm=llm,
        case=case,
        history=history,
        condition_name=condition_name,
        memory_profile=memory_profile,
    )

    result = run_turn(
        llm=llm,
        case=case,
        history=history,
        condition_name=condition_name,
        previous_memory=memory,
        memory_profile=memory_profile,
    )

    trajectory.append(
        {
            "step": 0,
            "history_before": history,
            "analysis": result["analysis"],
            "memory": result["memory"],
            "planner": result["planner"],
            "response": result["response"],
            "timings": result["timings"],
        }
    )

    return result, trajectory
