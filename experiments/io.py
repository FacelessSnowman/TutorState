import json
from pathlib import Path


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def completed_keys(
    path: Path,
) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()

    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            done.add(
                (
                    row["case_id"],
                    row["condition"],
                    row["seed"],
                )
            )
    return done