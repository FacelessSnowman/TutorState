"""Offline tests for the runner: adapters, memory warm-up, error handling, MRBench loading.

    python -m pytest tests/test_experiments.py
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.load import failure_rate, load_results
from experiments.dataloader import load_cases, parse_mrbench_history
from experiments.runner import run_experiment
from experiments.turn import build_prior_memory
from llm import BudgetExceeded

ROOT = Path(__file__).resolve().parent.parent


class FakeLLM:
    """Answers each agent by the shape of its JSON schema; counts calls per agent."""

    model = "fake"

    def __init__(self, planner_action="HINT", budget_after=None):
        self.planner_action = planner_action
        self.budget_after = budget_after
        self.calls = {"analyser": 0, "planner": 0, "generator": 0}

    def complete(self, system, user, json_schema=None):
        total = sum(self.calls.values())
        if self.budget_after is not None and total >= self.budget_after:
            raise BudgetExceeded("cap")
        props = (json_schema or {}).get("properties", {})
        if "error_type" in props:
            self.calls["analyser"] += 1
            return json.dumps({
                "reasoning": "r", "correct": False, "concept": "area", "error_type": "guess",
                "misconception": "gives a side length instead of the area", "confidence": "medium",
            })
        if "action" in props:
            self.calls["planner"] += 1
            return json.dumps({"reason": "r", "action": self.planner_action})
        self.calls["generator"] += 1
        return "What is the width?"


HISTORY = [
    {"role": "tutor", "text": "What is the area?"},
    {"role": "student", "text": "4"},
    {"role": "tutor", "text": "Not quite. Try again."},
    {"role": "student", "text": "4 m"},
]
CASE = {"case_id": "H_1", "history": HISTORY, "lesson_topic": "Area", "problem": None}


def test_memory_is_built_from_earlier_student_turns():
    llm = FakeLLM()
    memory, steps = build_prior_memory(llm=llm, case=CASE, history=HISTORY, condition_name="full")
    assert llm.calls["analyser"] == 1          # only the earlier student turn, not the final one
    assert steps[0]["history_before"] == HISTORY[:2]   # no later turns leak into the earlier analysis
    assert memory["attempts"] == 1 and memory["misconception_occurrences"] == 1


def test_conditions_without_memory_skip_the_warm_up():
    for condition in ("baseline", "analyser"):
        llm = FakeLLM()
        memory, steps = build_prior_memory(llm=llm, case=CASE, history=HISTORY, condition_name=condition)
        assert memory is None and steps == [] and llm.calls["analyser"] == 0


def _run(tmp_path, llm, conditions=("baseline", "full")):
    out = tmp_path / "run.jsonl"
    run_experiment(
        llm=llm, cases=[CASE, {**CASE, "case_id": "H_2"}], dataset_name="t", split="validation",
        conditions=list(conditions), seeds=[0], mode="teacher_forced",
        out_path=out, trajectory_path=tmp_path / "traj.jsonl",
    )
    return out


def test_full_condition_end_to_end(tmp_path):
    llm = FakeLLM()
    rows = load_results(_run(tmp_path, llm, conditions=("full",)))
    assert all(r["error"] is None for r in rows)
    assert rows[0]["memory"]["attempts"] == 2      # earlier turn + final turn
    assert rows[0]["planner"]["action"] == "HINT"
    assert llm.calls == {"analyser": 4, "planner": 2, "generator": 2}


def test_failed_case_is_recorded_and_the_run_continues(tmp_path):
    out = _run(tmp_path, FakeLLM(planner_action="LECTURE"))   # planner output invalid -> PlannerError
    rows = load_results(out)
    failed = [r for r in rows if r["error"]]
    assert {r["condition"] for r in failed} == {"full"} and len(failed) == 2
    assert failed[0]["error"]["type"] == "PlannerError" and failed[0]["response"] is None
    assert failure_rate(rows)["full"]["rate"] == 1.0
    assert failure_rate(rows)["baseline"]["rate"] == 0.0


def test_rerun_retries_failures_and_success_replaces_them(tmp_path):
    out = _run(tmp_path, FakeLLM(planner_action="LECTURE"))
    rerun = FakeLLM()
    _run(tmp_path, rerun)
    assert rerun.calls["planner"] == 2            # only the 2 failed full rows were redone
    rows = load_results(out)
    assert len(rows) == 4 and not any(r["error"] for r in rows)


def test_budget_exceeded_stops_the_run(tmp_path):
    with pytest.raises(BudgetExceeded):
        _run(tmp_path, FakeLLM(budget_after=1))


def test_parse_mrbench_history_joins_continuation_lines():
    turns = parse_mrbench_history("Tutor: Solve 2+2.\xa0\n\xa0Student: First,\n2+2 = 4\n\nso 4.\n")
    assert turns == [
        {"role": "tutor", "text": "Solve 2+2."},
        {"role": "student", "text": "First,\n2+2 = 4\nso 4."},
    ]


def test_mrbench_cases_load_with_conversation_ids():
    cases = load_cases(ROOT / "data" / "MrBench" / "test.json")
    raw = json.loads((ROOT / "data" / "MrBench" / "test.json").read_text(encoding="utf-8"))
    assert [c["case_id"] for c in cases] == [r["conversation_id"] for r in raw]
    assert all(c["history"][-1]["role"] == "student" for c in cases)
    assert all(c["reference_solution"] is None for c in cases)    # only with include_solution=True
    with_solution = load_cases(ROOT / "data" / "MrBench" / "test.json", include_solution=True)
    assert all(c["reference_solution"] for c in with_solution)


def test_bridge_still_loads_through_load_cases():
    cases = load_cases(ROOT / "data" / "Bridge" / "bridge_data_size_142" / "142_test.json")
    assert len(cases) == 22 and cases[0]["case_id"].startswith("H_")
