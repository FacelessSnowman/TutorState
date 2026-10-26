import random

from experiments.io import (
    append_jsonl,
    completed_keys,
)
from experiments.teacher_forced import (
    run_teacher_forced_case,
)
from experiments.free_running import (
    run_free_running_case,
)


def run_experiment(
    *,
    llm,
    cases,
    dataset_name,
    split,
    conditions,
    seeds,
    mode,
    out_path,
    trajectory_path,
    memory_profile=None,
    max_turns=4,
):

    done = completed_keys(out_path)

    total = (
        len(cases)
        * len(conditions)
        * len(seeds)
    )

    current = 0

    for case in cases:
        for condition in conditions:
            for seed in seeds:

                current += 1

                key = (
                    case["case_id"],
                    condition,
                    seed,
                )

                if key in done:
                    print(
                        f"[{current}/{total}] "
                        f"skip {key}"
                    )
                    continue

                print(
                    f"[{current}/{total}] "
                    f"run {key}"
                )

                random.seed(seed)

                if mode == "teacher_forced":

                    result, trajectory = (
                        run_teacher_forced_case(
                            llm=llm,
                            case=case,
                            condition_name=condition,
                            memory_profile=memory_profile,
                        )
                    )

                elif mode == "free_running":

                    result, trajectory = (
                        run_free_running_case(
                            llm=llm,
                            case=case,
                            condition_name=condition,
                            memory_profile=memory_profile,
                            max_turns=max_turns,
                        )
                    )

                else:
                    raise ValueError(
                        f"Unknown mode: {mode}"
                    )

                append_jsonl(
                    out_path,
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "case_id": case["case_id"],
                        "condition": condition,
                        "seed": seed,
                        "analysis": result["analysis"],
                        "memory": result["memory"],
                        "planner": result["planner"],
                        "response": result["response"],
                        "mode": mode,
                    },
                )

                for step in trajectory:
                    append_jsonl(
                        trajectory_path,
                        {
                            "dataset": dataset_name,
                            "split": split,
                            "case_id": case["case_id"],
                            "condition": condition,
                            "seed": seed,
                            "mode": mode,
                            **step,
                        },
                    )