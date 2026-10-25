"""Shared data contracts between the TutorState agents (proposal Tables 1-3).

DRAFT - field names must be agreed by the whole group before anyone builds on
them. Track C (planner + generator) only relies on:
  * Action (the enum)
  * the field names in Analysis and Memory (read, never written, by Track C)
  * PlannerOutput (written by the planner, read by the generator and eval)
"""

from enum import Enum
from typing import Literal, Optional, TypedDict


class Action(str, Enum):
    """Pedagogical action space (proposal Table 3). Values match BRIDGE z_what_clean."""

    PROBE = "PROBE"
    HINT = "HINT"
    EXPLAIN = "EXPLAIN"
    SIMPLIFY = "SIMPLIFY"
    CORRECT = "CORRECT"
    AFFIRM = "AFFIRM"
    ENCOURAGE = "ENCOURAGE"


ACTIONS = [a.value for a in Action]

# BRIDGE `e_clean` labels. The analyser's error_type must use exactly these so
# Track D can score it against the gold `e` field.
ErrorType = Literal[
    "guess", "right-idea", "diagnose", "careless", "imprecise", "misinterpret", "no_error"
]
Level = Literal["low", "medium", "high"]


class Turn(TypedDict):
    """One line of a BRIDGE conversation (c_h entry)."""

    role: Literal["student", "tutor"]
    text: str


class Analysis(TypedDict):
    """Student analyser output (proposal Table 2)."""

    correct: Optional[bool]          # None when the turn is not an attempt ("ok", "brb")
    concept: Optional[str]
    error_type: ErrorType
    misconception: Optional[str]     # free text, None if no specific misconception
    confidence: Level


class MisconceptionRecord(TypedDict):
    """One misconception the student has shown, kept even after the student moves on."""

    misconception: str
    concept: Optional[str]
    occurrences: int
    status: Literal["tentative", "recurring", "resolved"]   # seen once / seen 2+ times / later answered correctly


class Memory(TypedDict):
    """Persistent student state (proposal Table 1)."""

    current_concept: Optional[str]
    attempts: int
    correct_attempts: int
    current_misconception: Optional[str]
    misconception_occurrences: int
    hints_given: int
    mastery: Literal["unknown", "low", "medium", "high"]
    diagnosis_confidence: Level
    # Track B addition: every misconception seen so far with its own count, so a
    # misconception that comes back after a different one is recognised as
    # recurring instead of being counted from 1 again.
    misconception_history: list[MisconceptionRecord]


class PlannerOutput(TypedDict):
    """Pedagogical planner output. `reason` comes first so the model reasons before choosing."""

    reason: str
    action: str                      # always one of ACTIONS
