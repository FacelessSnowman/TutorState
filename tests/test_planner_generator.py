"""Offline tests for the planner and generator (no API key needed).

    python -m pytest tests/test_planner_generator.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.generator import GeneratorError, clean_response, generate
from agents.planner import PlannerError, plan
from llm import ScriptedLLM
from mocks.track_c import iter_mock_cases

CASE = next(iter_mock_cases())
INPUTS = CASE["inputs"]


def test_planner_returns_valid_action():
    llm = ScriptedLLM([{"reason": "First error on area.", "action": "hint"}])
    assert plan(llm, **INPUTS) == {"reason": "First error on area.", "action": "HINT"}


def test_planner_retries_then_succeeds():
    llm = ScriptedLLM(["not json", {"reason": "Unclear message.", "action": "PROBE"}])
    assert plan(llm, **INPUTS)["action"] == "PROBE"
    assert len(llm.calls) == 2


def test_planner_raises_instead_of_defaulting():
    llm = ScriptedLLM([{"reason": "x", "action": "LECTURE"}, {"reason": "x", "action": "PROBE/HINT"}])
    with pytest.raises(PlannerError) as err:
        plan(llm, **INPUTS)
    assert len(err.value.raw_outputs) == 2


def test_planner_sees_state_but_no_gold_labels():
    llm = ScriptedLLM([{"reason": "r", "action": "HINT"}])
    plan(llm, **INPUTS)
    user = llm.calls[0]["user"]
    assert "Diagnosis of the student's latest message" in user
    assert "Student state across the session" in user
    for expert in CASE["expected"]["expert_responses"]:
        assert expert not in user


def test_memory_lists_render_as_nested_bullets():
    llm = ScriptedLLM([{"reason": "r", "action": "EXPLAIN"}])
    repeated = next(c for c in iter_mock_cases() if c["name"] == "guess_area_repeated")
    plan(llm, **repeated["inputs"])
    user = llm.calls[0]["user"]
    assert "- misconception history:\n  - misconception: gives a side length" in user
    assert "status: recurring" in user
    assert "[{" not in user


def test_generator_ablation_only_changes_sections():
    """Baseline and full condition share the template; only the optional sections differ."""
    baseline_llm, full_llm = ScriptedLLM(["Hi"]), ScriptedLLM(["Hi"])
    generate(baseline_llm, INPUTS["history"], lesson_topic=INPUTS["lesson_topic"])
    generate(full_llm, **INPUTS, action="HINT", planner_reason="First error.")

    baseline, full = baseline_llm.calls[0], full_llm.calls[0]
    assert "Diagnosis" not in baseline["user"] and "Teaching move" not in baseline["user"]
    assert "Teaching move to make\nHINT\nWhy: First error." in full["user"]
    assert "Your teaching move for this message is HINT" in full["system"]
    assert full["system"].startswith(baseline["system"].rstrip("\n"))


def test_generator_rejects_unknown_action():
    with pytest.raises(ValueError):
        generate(ScriptedLLM(["Hi"]), INPUTS["history"], action="LECTURE")


LEAKED = (
    "The student has answered 2/2 twice now, and the tutor asked if it's final. I should gently check "
    "their thinking without giving away anything.\n\n"
    "Thanks for sticking with it! Before we decide if 2/2 is your final answer, can you tell me how you got it?"
)


def test_generator_drops_planning_notes_and_keeps_the_message():
    message = generate(ScriptedLLM([LEAKED]), INPUTS["history"])
    assert message.startswith("Thanks for sticking with it!")
    assert "the student" not in message.lower()


def test_generator_retries_when_the_reply_is_only_notes():
    llm = ScriptedLLM(["The student seems unsure, so I should probe.", "How did you get 4 m?"])
    assert generate(llm, INPUTS["history"]) == "How did you get 4 m?"
    assert len(llm.calls) == 2


def test_generator_raises_after_repeated_notes():
    llm = ScriptedLLM(["The student is unsure.", "I should ask the student a question."])
    with pytest.raises(GeneratorError):
        generate(llm, INPUTS["history"])


def test_generator_retries_a_reply_that_says_it_cannot_see_the_work():
    llm = ScriptedLLM(["I can't see your work on this one, so what did you get?", "How did you get 4 m?"])
    assert generate(llm, INPUTS["history"]) == "How did you get 4 m?"


def test_generator_keeps_a_cant_see_reply_rather_than_failing_the_case():
    reply = "The problem isn't shown here, so how did you get 4 m?"
    assert generate(ScriptedLLM([reply, reply]), INPUTS["history"]) == reply


def test_generator_prompt_forbids_saying_it_cannot_see_the_problem():
    llm = ScriptedLLM(["Hi"])
    generate(llm, INPUTS["history"])
    assert "Never say that you cannot see" in llm.calls[0]["system"]


@pytest.mark.parametrize("raw, cleaned", [
    ('Tutor: "What is the width?"', "What is the width?"),
    ("**Tutor:** Nice try!", "Nice try!"),
    ("“Good thinking.”", "Good thinking."),
    ("Let's look at the width.", "Let's look at the width."),
])
def test_clean_response(raw, cleaned):
    assert clean_response(raw) == cleaned
