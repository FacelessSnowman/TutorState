"""Run the planner + generator on the Track C mock cases with a real LLM.

    pip install anthropic
    set ANTHROPIC_API_KEY=...            (PowerShell: $env:ANTHROPIC_API_KEY="...")
    python scripts/try_planner_generator.py

Prints, per case, the planner decision next to the expert's, and the generated
message next to the expert's message. For eyeballing prompts, not for scoring.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.generator import generate
from agents.planner import plan
from llm import ClaudeLLM
from mocks.track_c import iter_mock_cases


def main():
    llm = ClaudeLLM()
    for case in iter_mock_cases():
        inputs, expected = case["inputs"], case["expected"]
        decision = plan(llm, **inputs)
        message = generate(llm, **inputs, action=decision["action"], planner_reason=decision["reason"])
        baseline = generate(llm, inputs["history"], lesson_topic=inputs["lesson_topic"])

        print(f"=== {case['name']}  (gold e={expected['e']}, gold action={expected['action']})")
        for turn in inputs["history"]:
            print(f"  {turn['role']:>7}: {turn['text']}")
        print(f"  planner : {decision['action']} - {decision['reason']}")
        print(f"  tutor   : {message}")
        print(f"  baseline: {baseline}")
        print(f"  expert  : {expected['expert_response']}\n")


if __name__ == "__main__":
    main()
