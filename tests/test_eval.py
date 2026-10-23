"""Offline tests for Track D scoring code (no API key needed).

    python -m pytest tests/test_eval.py
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.diagnosis import ERROR_TYPES, compare_diagnosis, score_diagnosis
from eval.human_sheet import compare_sheets, export_sheet, read_sheet, results_items, sheet_vs_judge
from eval.judge import JudgeError, judge, run_judge, run_judge_results
from eval.kappa import DIMENSIONS, agreement, to_score
from eval.leakage import final_answer, leakage_by_condition, reveals_answer
from eval.load import (
    load_bridge_contexts, load_bridge_gold, load_mrbench_contexts, load_mrbench_examples, load_results,
)
from eval.metrics import score
from eval.report import build_report
from eval.strategy import score_strategy
from llm import ScriptedLLM
from schemas import ACTIONS

GOLD = load_bridge_gold(ROOT / "data/Bridge/bridge_data_size_142/142_validation.json")
CASES = sorted(GOLD)


def wrong_label(refs, labels):
    return next(label for label in labels if label not in refs)


def fake_results(condition="full", seed=0, n_wrong=0):
    """Copy the first expert's labels, then make the first n_wrong cases wrong."""
    rows = []
    for i, case_id in enumerate(CASES):
        gold = GOLD[case_id]
        e, action = gold["e"][0], gold["actions"][0]
        if i < n_wrong:
            e, action = wrong_label(gold["e"], ERROR_TYPES), wrong_label(gold["actions"], ACTIONS)
        rows.append({
            "case_id": case_id, "condition": condition, "seed": seed,
            "analysis": {"error_type": e}, "planner": {"reason": "r", "action": action}, "response": "...",
        })
    return rows


# ---- metrics on dummy predictions ------------------------------------------

def test_copied_gold_scores_perfectly():
    report = score_strategy(fake_results(), GOLD)["strict"]["full"]
    assert report["any_match"]["mean"] == 1.0
    assert report["macro_f1"]["mean"] == 1.0


def test_six_wrong_out_of_21_gives_15_over_21():
    assert len(CASES) == 21
    report = score_diagnosis(fake_results(n_wrong=6), GOLD)["full"]
    assert report["any_match"]["mean"] == pytest.approx(15 / 21)


def test_matching_second_expert_counts_for_any_match_but_half_for_fraction():
    result = score(["EXPLAIN"], [["HINT", "EXPLAIN"]])
    assert result["any_match"] == 1.0
    assert result["fraction_matched"] == 0.5


def test_macro_f1_ignores_classes_absent_from_gold():
    result = score(["HINT", "PROBE"], [["HINT"], ["PROBE"]], labels=ACTIONS)
    assert result["macro_f1"] == 1.0
    assert result["labels"] == ACTIONS
    assert len(result["confusion"]) == len(ACTIONS)


def test_seeds_are_averaged_and_ci_brackets_mean():
    rows = fake_results(seed=0) + fake_results(seed=1, n_wrong=21)
    report = score_strategy(rows, GOLD)["strict"]["full"]
    assert report["seeds"] == [0, 1]
    assert report["any_match"]["mean"] == pytest.approx(0.5)
    lo, hi = report["any_match"]["ci"]
    assert lo <= 0.5 <= hi


def test_conditions_without_planner_are_skipped():
    baseline = [{**r, "condition": "baseline", "analysis": None, "planner": None} for r in fake_results()]
    report = score_strategy(baseline + fake_results(), GOLD)["strict"]
    assert set(report) == {"full"}


def test_lenient_set_accepts_extra_actions():
    rows = fake_results(n_wrong=21)
    acceptable = {r["case_id"]: [r["planner"]["action"]] for r in rows}
    report = score_strategy(rows, GOLD, acceptable)
    assert report["strict"]["full"]["any_match"]["mean"] == 0.0
    assert report["lenient"]["full"]["any_match"]["mean"] == 1.0


def test_paired_comparison_detects_better_condition():
    rows = fake_results("full") + fake_results("analyser_only", n_wrong=21)
    diff = compare_diagnosis(rows, GOLD, "full", "analyser_only")
    assert diff["diff"] == 1.0
    assert diff["ci"][0] > 0


def test_unknown_case_id_raises():
    rows = fake_results() + [{"case_id": "H_missing", "condition": "full", "seed": 0,
                              "analysis": {"error_type": "guess"}}]
    with pytest.raises(ValueError, match="no gold"):
        score_diagnosis(rows, GOLD)


def test_load_results_rejects_duplicates(tmp_path):
    path = tmp_path / "results.jsonl"
    row = json.dumps({"case_id": "H_1", "condition": "full", "seed": 0})
    path.write_text(row + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates"):
        load_results(path)


# ---- leakage ---------------------------------------------------------------

def test_final_answer_is_last_number_of_last_line():
    assert final_answer("3+4=7\n7-3= 4 balls\n 4") == 4
    assert final_answer("Total is $1,500.\n 1,500") == 1500
    assert final_answer(None) is None


def test_reveals_answer_normalises_formatting():
    solution = "...\n 1500"
    assert reveals_answer("So the desk costs $1,500.00 in total.", solution) is True
    assert reveals_answer("What do you get when you add 200 and 300?", solution) is False


def test_answer_already_in_conversation_cannot_be_checked():
    history = "Student: He had 2000 steps left, so 4000."
    assert reveals_answer("Where did you use those 2000 steps?", "...\n 2000", history) is None


def test_mrbench_solutions_all_have_a_final_answer():
    examples = load_mrbench_examples("train")
    assert all(final_answer(ex["ground_truth_solution"]) is not None for ex in examples)


# ---- kappa -----------------------------------------------------------------

def test_identical_raters_have_kappa_one():
    labels = ["Yes", "No", "To some extent", "Yes"]
    assert agreement(labels, labels, "Actionability")["kappa"] == pytest.approx(1.0)


def test_revealing_labels_collapse_to_binary():
    assert to_score("Revealing_of_the_Answer", "Yes (but the answer is incorrect)") == 1
    with pytest.raises(ValueError):
        to_score("Actionability", "Maybe")


# ---- judge (ScriptedLLM, no API calls) -------------------------------------

def ratings(label_for=lambda dim: "Yes"):
    from eval.kappa import SCALES
    return {dim: {"reasoning": "because", "label": label_for(dim) if label_for(dim) in SCALES[dim]
                  else next(iter(SCALES[dim]))} for dim in DIMENSIONS}


def test_judge_parses_and_never_sees_human_labels():
    example = load_mrbench_examples("train")[0]
    llm = ScriptedLLM([ratings()])
    result = judge(llm, example["conversation_history"], example["ground_truth_solution"],
                   example["candidate_response"])
    assert set(result) == set(DIMENSIONS)
    prompt = llm.calls[0]["user"]
    assert example["response_source"] not in prompt
    assert "human_ratings" not in prompt


def test_judge_retries_then_raises():
    llm = ScriptedLLM(["not json", {"Coherence": {"reasoning": "x", "label": "Yes"}}])
    with pytest.raises(JudgeError) as err:
        judge(llm, "Student: 5", "...\n 4", "Try again.")
    assert len(err.value.raw_outputs) == 2


def test_run_judge_skips_already_judged(tmp_path):
    examples = load_mrbench_examples("train")[:2]
    out = tmp_path / "judge.jsonl"
    assert run_judge(ScriptedLLM([ratings(), ratings()]), examples, out) == 2
    llm = ScriptedLLM([])
    assert run_judge(llm, examples, out) == 0
    assert llm.calls == []


def test_run_judge_results_hides_condition_and_shows_expert_context(tmp_path):
    contexts = load_bridge_contexts(ROOT / "data/Bridge/bridge_data_size_142/142_validation.json")
    rows = [{**r, "condition": "secret_condition"} for r in fake_results()[:2]]
    llm = ScriptedLLM([ratings(), ratings()])
    assert run_judge_results(llm, rows, contexts, tmp_path / "j.jsonl") == 2
    prompt = llm.calls[0]["user"]
    assert "secret_condition" not in prompt
    assert "Expert response (context only)" in prompt
    assert "Reference solution\nNot available." in prompt


# ---- report / leakage per condition / human sheets -------------------------

def mrbench_results():
    """Pretend two conditions answered MRBench train cases with existing candidate responses."""
    examples = load_mrbench_examples("train")
    by_case = {}
    for ex in examples:
        by_case.setdefault(ex["conversation_id"], []).append(ex["candidate_response"])
    return [
        {"case_id": cid, "condition": condition, "seed": 0, "response": responses[i]}
        for cid, responses in list(by_case.items())[:20]
        for i, condition in enumerate(["baseline", "full"])
    ]


def test_leakage_by_condition_counts_checked_and_unchecked():
    report = leakage_by_condition(mrbench_results(), load_mrbench_contexts("train"))
    for stats in report.values():
        assert stats["checked"] + stats["unchecked"] == 20
        assert stats["rate"] is None or 0 <= stats["rate"] <= 1


def test_build_report_runs_end_to_end(tmp_path):
    rows = fake_results("baseline", n_wrong=21) + fake_results("full", n_wrong=3)
    judge_rows = [{**{k: r[k] for k in ("case_id", "condition", "seed")}, "judge": ratings()} for r in rows]
    report = build_report(rows, ["baseline", "full"], gold=GOLD, judge_rows=judge_rows, out_dir=tmp_path)
    assert report["strategy_comparisons"]["full - baseline"]["diff"] == pytest.approx(18 / 21)
    assert report["judge"]["full"]["Coherence"]["mean"] == 2
    assert (tmp_path / "confusion_strategy_full.png").exists()
    json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))


def test_human_sheet_round_trip(tmp_path):
    contexts = load_bridge_contexts(ROOT / "data/Bridge/bridge_data_size_142/142_validation.json")
    rows = fake_results("cond_secret_a") + fake_results("cond_secret_b")
    sheet = tmp_path / "sheet.csv"
    key = export_sheet(results_items(rows, contexts), sheet, n=5)

    assert "cond_secret" not in sheet.read_text(encoding="utf-8-sig")  # blind to condition

    # Fill the sheet with "Yes"-style labels for two scorers, then compare.
    import csv
    with open(sheet, newline="", encoding="utf-8-sig") as f:
        filled = list(csv.DictReader(f))
    for row in filled:
        row.update({dim: v["label"] for dim, v in ratings().items()})
    for name in ("a.csv", "b.csv"):
        with open(tmp_path / name, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, list(filled[0]))
            writer.writeheader()
            writer.writerows(filled)
    agree = compare_sheets(tmp_path / "a.csv", tmp_path / "b.csv")
    assert agree["Coherence"]["exact_agreement"] == 1.0

    judge_rows = [{**{k: r[k] for k in ("case_id", "condition", "seed")}, "judge": ratings()} for r in rows]
    vs = sheet_vs_judge(tmp_path / "a.csv", key, judge_rows)
    assert vs["Coherence"]["n"] == 5


def test_partly_scored_sheet_row_raises(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("item_id,Coherence,Actionability\nitem001,Yes,\n", encoding="utf-8-sig")
    with pytest.raises(ValueError, match="partly"):
        read_sheet(path)
