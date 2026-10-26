"""Run the LLM judge (costs API credit). Always preview with --dry-run first.

Validate the judge on MRBench's own responses, then print kappa vs the human ratings:

    python -m eval.run_judge mrbench --split train --limit 50 --dry-run
    python -m eval.run_judge mrbench --split train --limit 50      # group model via OpenRouter (default)
    python -m eval.run_judge mrbench --split train --limit 50 --model <openrouter slug>   # another judge model

Judge TutorState responses from a Track E results file, then print mean scores per condition:

    python -m eval.run_judge results --results results/<file>.jsonl --bridge data/Bridge/bridge_data_size_142/142_validation.json --limit 20

The sample is random but fixed by --seed, so re-running with a larger --limit
reuses every response already judged (same prompt version) and only pays for
the new ones. Output goes to results/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from eval.judge import build_user_prompt, prompt_version, run_judge, run_judge_results, system_prompt
from eval.kappa import agreement_table
from eval.load import load_bridge_contexts, load_mrbench_contexts, load_mrbench_examples, load_results
from eval.report import judge_summary

PROVIDERS = ["openrouter", "anthropic", "openai"]  # openrouter = the group key and model (llm.get_llm)


def build_llm(provider: str, model: str | None, effort: str, max_tokens: int):
    """Same providers and defaults as Track E's runner, so the judge can use the same model."""
    from experiments.run_experiment import build_llm as build_experiment_llm

    return build_experiment_llm(
        argparse.Namespace(provider=provider, model=model, effort=effort, max_tokens=max_tokens)
    )


def sample(items: list, limit: int | None, seed: int) -> list:
    if limit is None or limit >= len(items):
        return items
    return random.Random(seed).sample(items, limit)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_env_file() -> None:
    """Read API keys from a local .env file (gitignored), if python-dotenv is installed.
    Keys already set in the terminal take precedence."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser(description="Run the LLM judge (costs API credit)")
    sub = parser.add_subparsers(dest="command", required=True)
    mrbench = sub.add_parser("mrbench", help="judge MRBench's own responses to validate the judge")
    mrbench.add_argument("--split", default="train", choices=["train", "validation", "test"])
    results = sub.add_parser("results", help="judge TutorState responses from a results file")
    results.add_argument("--results", required=True, type=Path)
    source = results.add_mutually_exclusive_group(required=True)
    source.add_argument("--bridge", help="Bridge split file the results were run on")
    source.add_argument("--mrbench-split", choices=["train", "validation", "test"])
    for p in (mrbench, results):
        p.add_argument("--limit", type=int, default=20, help="how many responses to judge (default 20)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--provider", choices=PROVIDERS, default="openrouter")
        p.add_argument("--model", default=None, help="model name; omit to use Track E's default")
        p.add_argument("--effort", choices=["low", "medium", "high"], default="low")
        p.add_argument("--max-tokens", type=int, default=4000)
        p.add_argument("--out", type=Path, default=None)
        p.add_argument("--dry-run", action="store_true", help="show what would run without calling the API")
    args = parser.parse_args()

    from experiments.run_experiment import DEFAULT_MODELS

    model = args.model or DEFAULT_MODELS[args.provider]
    model_tag = model.replace("/", "-").replace(":", "-")

    if args.command == "mrbench":
        items = sample(load_mrbench_examples(args.split), args.limit, args.seed)
        out = args.out or Path("results") / f"judge_mrbench_{args.split}_{model_tag}.jsonl"
        first = build_user_prompt(items[0]["conversation_history"], items[0]["ground_truth_solution"],
                                  items[0]["candidate_response"])
    else:
        rows = [r for r in load_results(args.results) if r.get("response")]
        contexts = load_bridge_contexts(args.bridge) if args.bridge else load_mrbench_contexts(args.mrbench_split)
        items = sample(rows, args.limit, args.seed)
        out = args.out or Path("results") / f"judge_{args.results.stem}_{model_tag}.jsonl"
        context = contexts[items[0]["case_id"]]
        first = build_user_prompt(context["conversation_history"], context["ground_truth_solution"],
                                  items[0]["response"], context["expert_response"])

    version = prompt_version()
    already = {r.get("prompt_version") for r in read_jsonl(out)}
    print(f"{len(items)} responses selected, model {args.provider}/{model}, prompt version {version}")
    print(f"output: {out}")
    print("responses already judged with this prompt version are skipped (no cost)" if version in already
          else "no earlier results for this prompt version: every selected response is one API call")

    if args.dry_run:
        print("\n--- system prompt ---\n" + system_prompt())
        print("\n--- first user prompt ---\n" + first)
        print("\ndry run: no API calls made")
        return

    llm = build_llm(args.provider, model, args.effort, args.max_tokens)
    if args.command == "mrbench":
        new = run_judge(llm, items, out)
    else:
        new = run_judge_results(llm, items, contexts, out)
    print(f"judged {new} new responses")
    usage = getattr(llm, "usage", None)
    if usage:
        print(f"LLM usage: {usage['calls']} calls, ${usage['cost_usd']:.4f}")

    judged = [r for r in read_jsonl(out) if r["prompt_version"] == version]
    if args.command == "mrbench":
        print(f"\nJudge vs human ratings ({len(judged)} responses, prompt {version}):")
        print(f"{'dimension':<26}{'n':>5}{'kappa':>8}{'exact':>8}")
        for dim, s in agreement_table(judged).items():
            print(f"{dim:<26}{s['n']:>5}{s['kappa']:>8.3f}{s['exact_agreement']:>8.3f}")
    else:
        print(f"\nMean judge scores per condition (0-2; Revealing_of_the_Answer 0-1):")
        for condition, dims in judge_summary(judged).items():
            print(f"  {condition}: " + ", ".join(f"{d}={v['mean']:.2f}" for d, v in dims.items()))


if __name__ == "__main__":
    main()
