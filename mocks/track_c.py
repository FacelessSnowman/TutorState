"""Mock inputs for developing the planner and generator before Tracks A and B land.

Conversation histories are real BRIDGE *train* cases (never test). The analysis
and memory dicts are hand-written stand-ins for the analyser and memory manager
output, in the schemas.py format. Gold labels are kept separately under
`expected` only so we can eyeball the planner's choice; they are never passed
to an agent.

Replace load_bridge_case with the Track A loader (data/bridge.py) once it exists.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_bridge_case(c_id: str, split: str = "train") -> dict:
    with open(DATA_DIR / f"bridge_{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["c_id"] == c_id:
                return record
    raise KeyError(f"{c_id} not in bridge_{split}.jsonl")


MOCKS = [
    {
        "name": "guess_area",  # student gives a side length when asked for an area
        "c_id": "4110653_19",
        "analysis": {
            "correct": False,
            "concept": "area of a rectangle",
            "error_type": "guess",
            "misconception": "gives a side length instead of computing length x width",
            "confidence": "medium",
        },
        "memory": {
            "current_concept": "area of a rectangle",
            "attempts": 1,
            "correct_attempts": 0,
            "current_misconception": "gives a side length instead of computing length x width",
            "misconception_occurrences": 1,
            "hints_given": 0,
            "mastery": "low",
            "diagnosis_confidence": "medium",
        },
    },
    {
        "name": "guess_area_repeated",  # same turn, but the misconception has recurred despite hints
        "c_id": "4110653_19",
        "analysis": {
            "correct": False,
            "concept": "area of a rectangle",
            "error_type": "guess",
            "misconception": "gives a side length instead of computing length x width",
            "confidence": "high",
        },
        "memory": {
            "current_concept": "area of a rectangle",
            "attempts": 4,
            "correct_attempts": 0,
            "current_misconception": "gives a side length instead of computing length x width",
            "misconception_occurrences": 3,
            "hints_given": 2,
            "mastery": "low",
            "diagnosis_confidence": "high",
        },
    },
    {
        "name": "careless_multiplication",  # 4 x 4 = 12
        "c_id": "2963623_41",
        "analysis": {
            "correct": False,
            "concept": "multiplication facts",
            "error_type": "careless",
            "misconception": "computed 4 x 3 instead of 4 x 4",
            "confidence": "high",
        },
        "memory": {
            "current_concept": "multiplication facts",
            "attempts": 1,
            "correct_attempts": 0,
            "current_misconception": None,
            "misconception_occurrences": 0,
            "hints_given": 0,
            "mastery": "medium",
            "diagnosis_confidence": "high",
        },
    },
    {
        "name": "unclear_answer",  # bare "0.8", problem not visible in the history
        "c_id": "2950057_118",
        "analysis": {
            "correct": None,
            "concept": "decimal numbers",
            "error_type": "diagnose",
            "misconception": None,
            "confidence": "low",
        },
        "memory": {
            "current_concept": "decimal numbers",
            "attempts": 1,
            "correct_attempts": 0,
            "current_misconception": None,
            "misconception_occurrences": 0,
            "hints_given": 0,
            "mastery": "unknown",
            "diagnosis_confidence": "low",
        },
    },
    {
        "name": "gibberish",  # "nhw"
        "c_id": "4055603_103",
        "analysis": {
            "correct": None,
            "concept": "multiples of ten",
            "error_type": "diagnose",
            "misconception": None,
            "confidence": "low",
        },
        "memory": {
            "current_concept": "multiples of ten",
            "attempts": 2,
            "correct_attempts": 0,
            "current_misconception": None,
            "misconception_occurrences": 0,
            "hints_given": 0,
            "mastery": "unknown",
            "diagnosis_confidence": "low",
        },
    },
]


def iter_mock_cases():
    """Yield agent inputs plus the gold labels (for eyeballing only)."""
    for mock in MOCKS:
        record = load_bridge_case(mock["c_id"])
        yield {
            "name": mock["name"],
            "inputs": {
                "history": record["c_h"],
                "lesson_topic": record["lesson_topic_clean"],
                "analysis": mock["analysis"],
                "memory": mock["memory"],
            },
            "expected": {
                "e": record["e_clean"],
                "action": record["z_what_clean"],
                "expert_response": " / ".join(r["text"] for r in record["c_r_"]),
            },
        }
