"""Run the planner + generator on the Track C mock cases with a real LLM.

    pip install openai
    put OPENROUTER_API_KEY=... in .env   (gitignored; never commit the key)
    python scripts/try_planner_generator.py

Prints, per case, the planner decision next to the expert's, and the generated
message next to the expert's message. For eyeballing prompts, not for scoring.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.generator import generate
from agents.planner import plan
from llm import get_llm
from mocks.track_c import iter_mock_cases


def main():
    llm = get_llm()
    for case in iter_mock_cases():
        inputs, expected = case["inputs"], case["expected"]
        decision = plan(llm, **inputs)
        message = generate(llm, **inputs, action=decision["action"], planner_reason=decision["reason"])
        baseline = generate(llm, inputs["history"], lesson_topic=inputs["lesson_topic"])

        print(f"=== {case['name']}  (gold e={expected['e']}, gold actions={expected['actions']})")
        for turn in inputs["history"]:
            print(f"  {turn['role']:>7}: {turn['text']}")
        print(f"  planner : {decision['action']} - {decision['reason']}")
        print(f"  tutor   : {message}")
        print(f"  baseline: {baseline}")
        for expert in expected["expert_responses"]:
            print(f"  expert  : {expert}")
        print()
    print(f"spent ${llm.usage['cost_usd']:.4f} over {llm.usage['calls']} calls")


if __name__ == "__main__":
    main()
