import argparse
from pathlib import Path

from experiments.conditions import CONDITIONS
from experiments.dataloader import load_bridge_cases
from experiments.runner import run_experiment
from llm import ClaudeLLM, GPTLLM


DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-6-astra",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TutorState experiments."
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the dataset split JSON file.",
    )

    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset identifier, e.g. bridge142.",
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "validation",
            "test",
        ],
        default="validation",
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(CONDITIONS),
        default=list(CONDITIONS),
        help="Ablation conditions to run.",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Experimental replicate identifiers.",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "teacher_forced",
            "free_running",
        ],
        default="teacher_forced",
    )

    parser.add_argument(
        "--memory-profile",
        default=None,
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Only run the first N cases. Useful for smoke tests.",
    )

    # ---------------------------------------------------------
    # LLM configuration
    # ---------------------------------------------------------

    parser.add_argument(
        "--provider",
        choices=[
            "anthropic",
            "openai",
        ],
        default="anthropic",
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name. If omitted, uses the default "
            "for the selected provider."
        ),
    )

    parser.add_argument(
        "--effort",
        choices=[
            "low",
            "medium",
            "high",
        ],
        default="medium",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16000,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--trajectory-out",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def build_llm(args):
    """
    Construct the requested LLM provider.

    Agents only interact with the shared LLM protocol, so nothing
    downstream needs to know whether this is Claude or GPT.
    """

    model = (
        args.model
        or DEFAULT_MODELS[args.provider]
    )

    if args.provider == "anthropic":
        return ClaudeLLM(
            model=model,
            effort=args.effort,
            max_tokens=args.max_tokens,
        )

    if args.provider == "openai":
        return GPTLLM(
            model=model,
            reasoning_effort=args.effort,
            max_tokens=args.max_tokens,
        )

    raise ValueError(
        f"Unknown provider: {args.provider}"
    )


def safe_name(value: str) -> str:
    """
    Make model/provider names safe for filenames.
    """

    return (
        value
        .replace("/", "-")
        .replace(":", "-")
        .replace(" ", "-")
    )


def main():
    args = parse_args()


    cases = load_bridge_cases(
        args.data
    )

    if args.max_cases is not None:
        cases = cases[:args.max_cases]


    model_name = (
        args.model
        or DEFAULT_MODELS[args.provider]
    )

    llm = build_llm(args)

    provider_name = safe_name(
        args.provider
    )

    model_file_name = safe_name(
        model_name
    )

    run_name = (
        f"{args.dataset_name}_"
        f"{args.split}_"
        f"{args.mode}_"
        f"{provider_name}_"
        f"{model_file_name}"
    )

    out = (
        args.out
        or Path(
            "results"
        )
        / f"{run_name}.jsonl"
    )

    trajectory_out = (
        args.trajectory_out
        or Path(
            "results"
        )
        / "trajectories"
        / f"{run_name}.jsonl"
    )
    
    run_experiment(
        llm=llm,
        cases=cases,
        dataset_name=args.dataset_name,
        split=args.split,
        conditions=args.conditions,
        seeds=args.seeds,
        mode=args.mode,
        out_path=out,
        trajectory_path=trajectory_out,
        memory_profile=args.memory_profile,
        max_turns=args.max_turns,
    )


if __name__ == "__main__":
    main()