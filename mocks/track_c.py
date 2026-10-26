"""Mock inputs for developing the planner and generator.

Conversation histories are real BRIDGE cases taken from bridge_data_size_441
*train* that are in neither version's test split nor 142's validation split
(the two versions were split independently). The analyses are hand-written
stand-ins for the analyser; the memories are built from them with Track B's
real memory manager, so they have exactly the shape the pipeline produces.

Gold labels are kept under `expected` only so we can eyeball the planner's
choice; they are never passed to an agent.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from agents.memory import build_memory
from eval.load import load_bridge_gold
from experiments.dataloader import load_bridge_cases

TRAIN_FILE = Path(__file__).resolve().parent.parent / "data" / "Bridge" / "bridge_data_size_441" / "441_train.json"

AREA = {
    "correct": False,
    "concept": "area of a rectangle",
    "error_type": "guess",
    "misconception": "gives a side length instead of computing length x width",
    "confidence": "medium",
}

MOCKS = [
    {
        "name": "guess_area",  # student gives a side length when asked for an area
        "history_id": "H_9e4bbaca044ad4d206fa",
        "earlier_analyses": [],
        "analysis": AREA,
    },
    {
        "name": "guess_area_repeated",  # same turn, but the misconception already recurred despite hints
        "history_id": "H_9e4bbaca044ad4d206fa",
        "earlier_analyses": [AREA, AREA],
        "hints_given": 2,
        "analysis": {**AREA, "confidence": "high"},
    },
    {
        "name": "careless_addition",  # asked 27 + 16, answers 11664
        "history_id": "H_475e65d1d77cdec9c089",
        "earlier_analyses": [],
        "analysis": {
            "correct": False,
            "concept": "two-digit addition",
            "error_type": "careless",
            "misconception": "multiplied instead of adding",
            "confidence": "medium",
        },
    },
    {
        "name": "unclear_answer",  # "yellow" to "What is the least favorite color?", chart not visible
        "history_id": "H_13765eac9abf4d94891a",
        "earlier_analyses": [],
        "analysis": {
            "correct": None,
            "concept": "reading a pictograph",
            "error_type": "diagnose",
            "misconception": None,
            "confidence": "low",
        },
    },
    {
        "name": "gibberish",  # "nhw"
        "history_id": "H_2d5a11f8f7d14c215d53",
        "earlier_analyses": [],
        "analysis": {
            "correct": None,
            "concept": "multiples of ten",
            "error_type": "diagnose",
            "misconception": None,
            "confidence": "low",
        },
    },
]


@lru_cache(maxsize=None)
def _cases() -> dict:
    return {case["case_id"]: case for case in load_bridge_cases(TRAIN_FILE)}


@lru_cache(maxsize=None)
def _gold() -> dict:
    return load_bridge_gold(TRAIN_FILE)


def iter_mock_cases():
    """Yield agent inputs plus the gold labels (for eyeballing only)."""
    for mock in MOCKS:
        case = _cases()[mock["history_id"]]
        memory = build_memory(mock["earlier_analyses"] + [mock["analysis"]], history=case["history"])
        if "hints_given" in mock:
            memory["hints_given"] = mock["hints_given"]
        gold = _gold()[mock["history_id"]]
        yield {
            "name": mock["name"],
            "inputs": {
                "history": case["history"],
                "lesson_topic": case["lesson_topic"],
                "analysis": mock["analysis"],
                "memory": memory,
            },
            "expected": {
                "e": gold["e"],
                "actions": gold["actions"],
                "expert_responses": gold["expert_responses"],
            },
        }
