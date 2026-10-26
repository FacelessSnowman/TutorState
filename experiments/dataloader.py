import json
import re
from pathlib import Path

# A new MRBench turn starts with "Tutor:" or "Student:" at the start of a line;
# any other line continues the previous turn (multi-line working, equations).
_MRBENCH_SPEAKER = re.compile(r"^\s*(Tutor|Student)\s*:\s*", re.IGNORECASE)


def load_cases(path: Path, include_solution: bool = False) -> list[dict]:
    """Load BRIDGE or MRBench cases, detected from the file's record format."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    if rows and "conversation_id" in rows[0]:
        return load_mrbench_cases(path, include_solution=include_solution)
    return load_bridge_cases(path)


def parse_mrbench_history(text: str) -> list[dict]:
    """Split MRBench's single conversation string into [{"role", "text"}] turns."""
    turns: list[dict] = []
    for raw_line in text.replace("\xa0", " ").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _MRBENCH_SPEAKER.match(line)
        if match:
            turns.append({"role": match.group(1).lower(), "text": line[match.end():].strip()})
        elif turns:
            turns[-1]["text"] += "\n" + line
        else:
            raise ValueError(f"conversation does not start with a speaker: {line[:60]!r}")
    return turns


def load_mrbench_cases(path: Path, include_solution: bool = False) -> list[dict]:
    """MRBench conversations as runner cases.

    case_id is conversation_id, which is the key Track D's leakage check and
    judge use. The problem statement is inside the first tutor turn, so
    `problem` stays None. The reference solution is only passed to the agents
    when include_solution is True (a group decision: it grounds the tutor, but
    MRBench's own tutors were judged without it). Conversations that do not
    end with a student turn cannot be answered by the tutor and are skipped.
    """
    cases = []
    skipped = []

    for row in json.loads(path.read_text(encoding="utf-8")):
        history = parse_mrbench_history(row["conversation_history"])
        if not history or history[-1]["role"] != "student":
            skipped.append(row["conversation_id"])
            continue

        solution = row.get("Ground_Truth_Solution")
        if isinstance(solution, str) and solution.strip().casefold() in ("", "not available"):
            solution = None

        cases.append(
            {
                "case_id": row["conversation_id"],
                "history": history,
                "lesson_topic": None,
                "problem": None,
                "reference_solution": solution if include_solution else None,
            }
        )

    if skipped:
        print(f"MRBench: skipped {len(skipped)} conversation(s) not ending with a student turn: {skipped}")
    return cases


def load_bridge_cases(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))

    cases = []

    for row in rows:
        history = [
            {
                "role": turn["role"],
                "text": turn["text"],
            }
            for turn in row["c_h"]
        ]

        if not history:
            raise ValueError(
                f"{row['history_id']} has empty history"
            )

        if history[-1]["role"] != "student":
            raise ValueError(
                f"{row['history_id']} must end with a student turn"
            )

        cases.append(
            {
                "case_id": row["history_id"],
                "history": history,
                "lesson_topic": row.get("lesson_topic"),
                "problem": None,
            }
        )

    return cases