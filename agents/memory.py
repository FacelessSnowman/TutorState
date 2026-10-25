"""Memory manager (Track B): keeps the student state across turns.

    update_memory(previous_memory, analysis, history) -> Memory (schemas.py, proposal Table 1)

Plain Python, no LLM call: every update is a deterministic rule, so it is cheap,
reproducible across seeds, and easy to check in the memory-consistency tests.

Rules
-----
* Attempts: every analysed message that is an actual attempt (correct is True
  or False) adds one attempt; a correct one also adds one correct attempt.
  Non-attempts ("ok", "im done", correct is None) change no counts.
* Recurring vs one-off misconception (proposal 3.1): a misconception seen once
  is "tentative" (weak evidence, maybe a slip); seen two or more times it is
  "recurring" (strong evidence). Every misconception is kept in
  misconception_history with its own count, so if the student makes mistake A,
  then B, then A again, A is recognised as seen twice.
* A correct answer on a concept marks that concept's misconceptions "resolved"
  and clears the current one. The count is kept, so if the mistake comes back
  it goes straight to recurring.
* Two misconception descriptions count as the same if their words overlap
  enough (same_misconception). This is a simple heuristic: wording that differs
  a lot ("adds bottoms" vs "adds denominators") will be treated as different.
* hints_given is counted from the tutor messages in the history (count_hints).
  Also a heuristic: there is no hint label in BRIDGE.
* Mastery comes from the correct-attempt rate, and is low while a recurring
  misconception is still active.
"""

from __future__ import annotations

import copy
import re
from typing import Optional, Sequence

from schemas import Analysis, Memory, MisconceptionRecord, Turn

RECURRING_AT = 2          # occurrences needed to treat a misconception as recurring
SAME_MISCONCEPTION_AT = 0.6  # word overlap needed to treat two descriptions as the same

_STOPWORDS = {
    "a", "an", "the", "of", "to", "and", "or", "in", "on", "for", "with", "by", "is", "it",
    "instead", "student", "students", "their", "they", "them", "its", "as", "at", "be", "than",
}
_HINT_PATTERN = re.compile(
    r"\b(hint|remember|think about|try to|first step|we need to|you need to|start by|look at)\b",
    re.IGNORECASE,
)


def empty_memory() -> Memory:
    return {
        "current_concept": None,
        "attempts": 0,
        "correct_attempts": 0,
        "current_misconception": None,
        "misconception_occurrences": 0,
        "hints_given": 0,
        "mastery": "unknown",
        "diagnosis_confidence": "low",
        "misconception_history": [],
    }


def _words(text: Optional[str]) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in tokens if t not in _STOPWORDS}


def same_misconception(a: str, b: str) -> bool:
    """True if two misconception descriptions share enough words (Jaccard overlap)."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= SAME_MISCONCEPTION_AT


def same_concept(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    wa, wb = _words(a), _words(b)
    return bool(wa) and bool(wb) and len(wa & wb) / len(wa | wb) >= 0.5


def count_hints(history: Optional[Sequence[Turn]]) -> int:
    """Tutor messages that look like guidance (heuristic keyword match)."""
    if not history:
        return 0
    return sum(1 for t in history if t["role"] == "tutor" and _HINT_PATTERN.search(t["text"]))


def _status(occurrences: int) -> str:
    return "recurring" if occurrences >= RECURRING_AT else "tentative"


def _mastery(memory: Memory) -> str:
    attempts, correct = memory["attempts"], memory["correct_attempts"]
    if attempts == 0:
        return "unknown"
    active_recurring = any(
        r["status"] == "recurring" and r["misconception"] == memory["current_misconception"]
        for r in memory["misconception_history"]
    )
    if active_recurring:
        return "low"
    rate = correct / attempts
    if rate >= 0.75 and attempts >= 2:
        return "high"
    if rate >= 0.5:
        return "medium"
    return "low"


def update_memory(
    previous_memory: Optional[Memory],
    analysis: Optional[Analysis],
    history: Optional[Sequence[Turn]] = None,
) -> Memory:
    """Return the new memory after one analysed student message. Never modifies previous_memory."""
    memory: Memory = copy.deepcopy(previous_memory) if previous_memory else empty_memory()
    memory.setdefault("misconception_history", [])

    if history is not None:
        memory["hints_given"] = count_hints(history)
    if analysis is None:
        return memory

    if analysis.get("concept"):
        memory["current_concept"] = analysis["concept"]
    memory["diagnosis_confidence"] = analysis["confidence"]

    correct = analysis.get("correct")
    if correct is None:                      # not an attempt: no counts change
        return memory

    memory["attempts"] += 1
    records: list[MisconceptionRecord] = memory["misconception_history"]

    if correct:
        memory["correct_attempts"] += 1
        for record in records:               # showed understanding of this concept
            if record["status"] != "resolved" and same_concept(record["concept"], analysis.get("concept")):
                record["status"] = "resolved"
        memory["current_misconception"] = None
        memory["misconception_occurrences"] = 0

    elif analysis.get("misconception"):
        text = analysis["misconception"]
        record = next((r for r in records if same_misconception(r["misconception"], text)), None)
        if record is None:
            record = {"misconception": text, "concept": analysis.get("concept"), "occurrences": 0, "status": "tentative"}
            records.append(record)
        record["occurrences"] += 1
        record["status"] = _status(record["occurrences"])
        memory["current_misconception"] = record["misconception"]
        memory["misconception_occurrences"] = record["occurrences"]
        if record["status"] == "recurring":  # repeated evidence makes the diagnosis firmer
            memory["diagnosis_confidence"] = "high"
    # A wrong answer with no identifiable misconception keeps the current one as is.

    memory["mastery"] = _mastery(memory)
    return memory


def build_memory(
    analyses: Sequence[Optional[Analysis]],
    history: Optional[Sequence[Turn]] = None,
    previous_memory: Optional[Memory] = None,
) -> Memory:
    """Fold a sequence of analyses (one per student turn, oldest first) into one memory."""
    memory = previous_memory
    for analysis in analyses:
        memory = update_memory(memory, analysis)
    return update_memory(memory, None, history)
