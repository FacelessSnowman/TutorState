import json
from pathlib import Path


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