"""Shared LLM client used by every agent.

DRAFT - the group owns this file. Agents only depend on the `LLM` protocol, so
swapping provider or model never touches agent code.
"""

from __future__ import annotations

import json
from typing import Optional, Protocol


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def complete(self, system: str, user: str, json_schema: Optional[dict] = None) -> str:
        """Return the model's text. With json_schema, the text is a JSON string matching it."""
        ...


class ClaudeLLM:
    """Claude via the Anthropic SDK (`pip install anthropic`, needs ANTHROPIC_API_KEY).

    Claude Opus 5.5 does not accept `temperature`; run-to-run variation is its
    natural sampling, and `effort` (low/medium/high/xhigh/max) controls thinking depth.
    """

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", max_tokens: int = 16000):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str, json_schema: Optional[dict] = None) -> str:
        output_config: dict = {"effort": self.effort}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
        )
        if response.stop_reason == "refusal":
            raise LLMError(f"refused: {response.stop_details}")
        if response.stop_reason == "max_tokens":
            raise LLMError("hit max_tokens before finishing")
        return "".join(b.text for b in response.content if b.type == "text")


class ScriptedLLM:
    """Offline stand-in for tests: returns queued replies and records every prompt."""

    def __init__(self, replies: list):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, system: str, user: str, json_schema: Optional[dict] = None) -> str:
        self.calls.append({"system": system, "user": user, "json_schema": json_schema})
        reply = self.replies.pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)
