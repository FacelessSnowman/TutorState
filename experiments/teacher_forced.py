import copy

from experiments.turn import run_turn


def run_teacher_forced_case(
    *,
    llm,
    case,
    condition_name,
    memory_profile=None,
):
    history = copy.deepcopy(case["history"])

    result = run_turn(
        llm=llm,
        case=case,
        history=history,
        condition_name=condition_name,
        previous_memory=None,
        memory_profile=memory_profile,
    )

    trajectory = [
        {
            "step": 0,
            "history_before": history,
            "analysis": result["analysis"],
            "memory": result["memory"],
            "planner": result["planner"],
            "response": result["response"],
            "timings": result["timings"],
        }
    ]

    return result, trajectory