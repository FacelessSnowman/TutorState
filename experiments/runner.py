import random
from collections import Counter

from agents.analyser import AnalyserError
from agents.generator import GeneratorError
from agents.planner import PlannerError
from llm import BudgetExceeded, LLMError

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


def _case_failure_types() -> tuple:
    """Errors that mean "this case failed", as opposed to a bug in our code.

    Agent errors (invalid output after retries), LLM errors (refusal, empty or
    truncated output) and provider API errors (rate limits, timeouts, 5xx) are
    recorded and the run moves on. Anything else - KeyError, TypeError,
    ValueError - is a bug and still crashes, so it gets fixed instead of being
    silently counted as a failure on every case.
    """
    types = [AnalyserError, PlannerError, GeneratorError, LLMError]
    for module in ("openai", "anthropic"):
        try:
            types.append(__import__(module).APIError)
        except ImportError:
            pass
    return tuple(types)


CASE_FAILURES = _case_failure_types()


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

    # Only successful rows count as done, so a rerun retries earlier failures.
    done = completed_keys(out_path)

    total = (
        len(cases)
        * len(conditions)
        * len(seeds)
    )

    current = 0
    outcomes = Counter()

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

                row = {
                    "dataset": dataset_name,
                    "split": split,
                    "case_id": case["case_id"],
                    "condition": condition,
                    "seed": seed,
                    "mode": mode,
                    "model": getattr(llm, "model", None),
                }

                try:
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

                except BudgetExceeded:
                    # Out of money: stop now. Finished rows are already saved.
                    _print_summary(outcomes)
                    raise

                except CASE_FAILURES as error:
                    outcomes[(condition, "failed")] += 1
                    print(
                        f"    FAILED {type(error).__name__}: "
                        f"{str(error)[:200]}"
                    )
                    append_jsonl(
                        out_path,
                        {
                            **row,
                            "analysis": None,
                            "memory": None,
                            "planner": None,
                            "response": None,
                            "error": {
                                "type": type(error).__name__,
                                "message": str(error)[:1000],
                            },
                        },
                    )
                    continue

                outcomes[(condition, "ok")] += 1

                append_jsonl(
                    out_path,
                    {
                        **row,
                        "analysis": result["analysis"],
                        "memory": result["memory"],
                        "planner": result["planner"],
                        "response": result["response"],
                        "error": None,
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

    _print_summary(outcomes)


def _print_summary(outcomes: Counter) -> None:
    conditions = sorted({condition for condition, _ in outcomes})
    if not conditions:
        return
    print("\nRun summary (this session):")
    for condition in conditions:
        ok = outcomes[(condition, "ok")]
        failed = outcomes[(condition, "failed")]
        print(
            f"  {condition:<10} ok={ok} failed={failed} "
            f"failure rate={failed / (ok + failed):.1%}"
        )
    if any(status == "failed" for _, status in outcomes):
        print("  Rerun the same command to retry failed cases.")
