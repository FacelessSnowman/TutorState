"""Build development files that never touch any candidate reporting set.

The 142 and 441 BRIDGE versions were split independently, so the same history
can be in one version's train/validation and the other's test. This writes
441 train and validation with every history from 142 validation, 142 test and
441 test removed, so prompt tuning and dev runs stay clean whichever of those
the group reports on.

    python scripts/make_dev_set.py
    -> results/dev/dev_train.json, results/dev/dev_validation.json  (gitignored; rerun to rebuild)
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRIDGE = ROOT / "data" / "Bridge"
OUT = ROOT / "results" / "dev"

REPORTING_SETS = [("142", "validation"), ("142", "test"), ("441", "test")]


def load(version: str, split: str) -> list[dict]:
    path = BRIDGE / f"bridge_data_size_{version}" / f"{version}_{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    blocked = {row["history_id"] for v, s in REPORTING_SETS for row in load(v, s)}
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        rows = load("441", split)
        kept = [row for row in rows if row["history_id"] not in blocked]
        path = OUT / f"dev_{split}.json"
        path.write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{path.relative_to(ROOT)}: {len(kept)} histories ({len(rows) - len(kept)} removed)")


if __name__ == "__main__":
    main()
