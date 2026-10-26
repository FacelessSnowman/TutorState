"""Show how much of the OpenRouter key's budget has been spent (whole group, all runs).

    python scripts/check_credits.py

Uses whatever key get_llm() would use: OPENROUTER_API_KEY from the environment,
else .env. The key itself is never printed.
"""

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm import _env


def main():
    key = _env("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set (environment or .env)")
    request = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"})
    data = json.load(urllib.request.urlopen(request, timeout=20))["data"]

    limit, used = data.get("limit"), data.get("usage", 0.0)
    print(f"spent:      ${used:.4f}  (today ${data.get('usage_daily', 0):.4f}, this week ${data.get('usage_weekly', 0):.4f})")
    if limit is not None:
        print(f"limit:      ${limit:.2f}  -> remaining ${data.get('limit_remaining', limit - used):.4f}")
    else:
        print("limit:      none set on this key")
    free = data.get("free_model_daily_requests")
    if free:
        print(f"free-model requests today: {free['used']} / {free['limit']}")


if __name__ == "__main__":
    main()
