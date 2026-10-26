"""Offline tests for the analyser and memory manager (no API key needed).

    python -m pytest tests/test_analyser_memory.py
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.analyser import ERROR_TYPES, AnalyserError, analyse, system_prompt
from agents.memory import build_memory, count_hints, empty_memory, same_misconception, update_memory
from agents.planner import plan
from llm import ScriptedLLM

HISTORY = [
    {"role": "tutor", "text": "Finally, we need to add the areas of both rectangles."},
    {"role": "tutor", "text": "What is the value of 36+42?"},
    {"role": "student", "text": "1,512"},
]
REPLY = {
    "reasoning": "36+42=78; 1512 is 36x42, so the student multiplied.",
    "correct": False,
    "concept": "Addition",
    "error_type": "Careless",
    "misconception": "multiplied instead of added",
    "confidence": "HIGH",
}


def analysis(correct, misconception=None, concept="fraction addition", error_type="careless", confidence="medium"):
    return {"correct": correct, "concept": concept, "error_type": error_type,
            "misconception": misconception, "confidence": confidence}


# ---------------------------------------------------------------- analyser

def test_analyser_returns_table2_fields():
    result = analyse(ScriptedLLM([REPLY]), HISTORY, lesson_topic="Areas by Decomposition")
    assert result == {
        "correct": False,
        "concept": "Addition",
        "error_type": "careless",
        "misconception": "multiplied instead of added",
        "confidence": "high",
    }


def test_analyser_can_keep_reasoning_for_debugging():
    result = analyse(ScriptedLLM([REPLY]), HISTORY, include_reasoning=True)
    assert result["reasoning"].startswith("36+42=78")


def test_analyser_retries_then_raises_instead_of_defaulting():
    llm = ScriptedLLM(["not json", {**REPLY, "error_type": "confused"}])
    with pytest.raises(AnalyserError) as err:
        analyse(llm, HISTORY)
    assert len(err.value.raw_outputs) == 2


def test_analyser_retry_succeeds():
    llm = ScriptedLLM(["not json", REPLY])
    assert analyse(llm, HISTORY)["error_type"] == "careless"
    assert len(llm.calls) == 2


def test_no_error_means_correct():
    reply = {**REPLY, "correct": False, "error_type": "no_error", "misconception": "none"}
    result = analyse(ScriptedLLM([reply]), HISTORY)
    assert result["correct"] is True and result["misconception"] is None


def test_non_attempt_keeps_correct_null():
    reply = {**REPLY, "correct": None, "error_type": "diagnose", "misconception": None, "confidence": "low"}
    assert analyse(ScriptedLLM([reply]), HISTORY)["correct"] is None


def test_prompt_lists_all_labels_and_sends_schema():
    llm = ScriptedLLM([REPLY])
    analyse(llm, HISTORY, lesson_topic="Areas by Decomposition")
    call = llm.calls[0]
    for label in ERROR_TYPES:
        assert f"- {label}:" in system_prompt()
    assert call["json_schema"]["properties"]["error_type"]["enum"] == ERROR_TYPES
    assert "1,512" in call["user"] and "Areas by Decomposition" in call["user"]


def test_analyser_never_sees_gold_labels():
    case = json.loads((Path(__file__).resolve().parent.parent
                       / "data/Bridge/bridge_data_size_142/142_train.json").read_text(encoding="utf-8"))[0]
    llm = ScriptedLLM([REPLY])
    analyse(llm, case["c_h"], lesson_topic=case["lesson_topic"])
    prompt = llm.calls[0]["system"] + llm.calls[0]["user"]
    for ann in case["annotations"]:
        for turn in ann["c_r_"]:
            assert turn["text"] not in prompt
    assert "e_clean" not in prompt and "z_what" not in prompt


# ---------------------------------------------------------------- memory

def test_proposal_worked_example_tables_5_to_6():
    """Proposal Table 5 -> Table 6: the 1/2 + 1/3 = 2/5 turn."""
    before = {**empty_memory(), "current_concept": "fraction addition", "attempts": 1, "mastery": "low"}
    after = update_memory(before, analysis(False, "adds numerators and denominators independently", confidence="high"))
    assert after["attempts"] == 2
    assert after["correct_attempts"] == 0
    assert after["current_misconception"] == "adds numerators and denominators independently"
    assert after["misconception_occurrences"] == 1
    assert after["mastery"] == "low"
    assert after["misconception_history"][0]["status"] == "tentative"
    assert before["attempts"] == 1          # previous memory not modified


def test_one_off_vs_recurring():
    m = update_memory(None, analysis(False, "adds the denominators"))
    assert m["misconception_history"][0]["status"] == "tentative"
    m = update_memory(m, analysis(False, "adds the denominators together"))
    assert m["misconception_occurrences"] == 2
    assert m["misconception_history"][0]["status"] == "recurring"
    assert m["diagnosis_confidence"] == "high"


def test_mistake_coming_back_after_a_different_one_is_recurring():
    """A, then B, then A again: A must be counted twice (one current_misconception slot would lose it)."""
    m = update_memory(None, analysis(False, "adds the denominators"))
    m = update_memory(m, analysis(False, "forgets to carry the one", concept="addition"))
    assert m["current_misconception"] == "forgets to carry the one"
    m = update_memory(m, analysis(False, "adds the denominators"))
    assert m["current_misconception"] == "adds the denominators"
    assert m["misconception_occurrences"] == 2
    assert len(m["misconception_history"]) == 2


def test_correct_answer_resolves_but_remembers():
    m = update_memory(None, analysis(False, "adds the denominators"))
    m = update_memory(m, analysis(True, error_type="no_error"))
    assert m["current_misconception"] is None
    assert m["misconception_history"][0]["status"] == "resolved"
    m = update_memory(m, analysis(False, "adds the denominators"))
    assert m["misconception_history"][0]["status"] == "recurring"   # came back, so recurring


def test_non_attempt_changes_no_counts():
    m = update_memory(None, analysis(False, "adds the denominators"))
    m2 = update_memory(m, analysis(None, error_type="diagnose", confidence="low"))
    assert m2["attempts"] == m["attempts"] == 1


def test_mastery_follows_correct_rate():
    m = build_memory([analysis(True, error_type="no_error")] * 3)
    assert m["attempts"] == 3 and m["mastery"] == "high"
    m = build_memory([analysis(False, "adds the denominators")] * 2)
    assert m["mastery"] == "low"
    assert build_memory([])["mastery"] == "unknown"


def test_same_misconception_matching():
    assert same_misconception("adds the denominators", "adds denominators together")
    assert not same_misconception("adds the denominators", "forgets to carry")


def test_hints_counted_from_tutor_messages():
    history = [
        {"role": "tutor", "text": "Let me try to provide a hint."},
        {"role": "tutor", "text": "Remember, we need to subtract."},
        {"role": "student", "text": "remember?"},
        {"role": "tutor", "text": "What is 10 minus 6?"},
        {"role": "student", "text": "5"},
    ]
    assert count_hints(history) == 2


# ---------------------------------------------------------------- integration with Track C

def test_planner_accepts_analyser_and_memory_output():
    a = analyse(ScriptedLLM([REPLY]), HISTORY)
    m = update_memory(None, a, HISTORY)
    llm = ScriptedLLM([{"reason": "Careless slip on addition.", "action": "CORRECT"}])
    assert plan(llm, HISTORY, analysis=a, memory=m)["action"] == "CORRECT"
    assert "multiplied instead of added" in llm.calls[0]["user"]
