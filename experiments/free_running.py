import copy

from experiments.adapters import run_student_simulator
from experiments.turn import build_prior_memory, run_turn


def run_free_running_case(
    *,
    llm,
    case,
    condition_name,
    memory_profile=None,
    max_turns=4,
):
    """
    runs free until we hit max turns iterations, there is technially no ending
    """
    history = copy.deepcopy(case["history"])

    # Start from the memory the dataset's earlier student turns imply.
    previous_memory, trajectory = build_prior_memory(
        llm=llm,
        case=case,
        history=history,
        condition_name=condition_name,
        memory_profile=memory_profile,
    )
    final_result = None

    for step in range(max_turns):

        result = run_turn(
            llm=llm,
            case=case,
            history=history,
            condition_name=condition_name,
            previous_memory=previous_memory,
            memory_profile=memory_profile,
        )

        final_result = result

        trajectory.append(
            {
                "step": step,
                "history_before": copy.deepcopy(history),
                "analysis": result["analysis"],
                "memory": result["memory"],
                "planner": result["planner"],
                "response": result["response"],
                "timings": result["timings"],
            }
        )

        previous_memory = result["memory"]

        if step == max_turns - 1:
            break

        history.append(
            {
                "role": "tutor",
                "text": result["response"],
            }
        )

        student_response = run_student_simulator(
            llm=llm,
            history=history,
            case=case,
            step=step,
        )

        history.append(
            {
                "role": "student",
                "text": student_response,
            }
        )

        trajectory[-1][
            "student_response"
        ] = student_response

    return final_result, trajectory