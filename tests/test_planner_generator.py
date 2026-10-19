"""Offline tests for the planner and generator (no API key needed).

    python -m pytest tests/test_planner_generator.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.generator import clean_response, generate
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
    assert CASE["expected"]["expert_response"] not in user


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


@pytest.mark.parametrize("raw, cleaned", [
    ('Tutor: "What is the width?"', "What is the width?"),
    ("**Tutor:** Nice try!", "Nice try!"),
    ("“Good thinking.”", "Good thinking."),
    ("Let's look at the width.", "Let's look at the width."),
])
def test_clean_response(raw, cleaned):
    assert clean_response(raw) == cleaned
