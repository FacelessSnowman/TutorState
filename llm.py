"""Shared LLM client used by every agent.

DRAFT - the group owns this file. Agents only depend on the `LLM` protocol, so
swapping provider or model never touches agent code.
"""

from __future__ import annotations

import json
from typing import Optional, Protocol
import os


# ---------------------------------------------------------------------------
# Group standard: every agent, run and judge uses this, via get_llm().
# Chosen 2026-10-26 from a 24-case bake-off on BRIDGE train: Claude Haiku 5.5
# was the most reliable (24/24 valid JSON) and joint-best on diagnosis and
# action accuracy, at $0.10 / $0.50 per million tokens (~$0.0002 per call).
# Changing these changes everyone's results - agree it with the group first.
# ---------------------------------------------------------------------------
MODEL = "anthropic/claude-haiku-5.5"
EFFORT = "low"                 # reasoning effort; low keeps output tokens (the expensive part) small
MAX_TOKENS = 4000              # per call, reasoning included; caps a single call at ~$0.002
MAX_SPEND_USD = 3.0            # per process; stops a runaway loop from eating the $15 group budget


class LLMError(RuntimeError):
    pass


class BudgetExceeded(LLMError):
    """The per-process spend cap was hit. Runners must stop, not skip the case."""


def get_llm(**overrides) -> "OpenRouterLLM":
    """The group-standard LLM. Use this everywhere instead of constructing a client directly."""
    return OpenRouterLLM(**overrides)


class LLM(Protocol):
    def complete(self, system: str, user: str, json_schema: Optional[dict] = None) -> str:
        """Return the model's text. With json_schema, the text is a JSON string matching it."""
        ...


class ClaudeLLM:
    """Claude via the Anthropic SDK directly. Not the group standard - use get_llm().

    Needs `pip install anthropic` and an ANTHROPIC_API_KEY with API credits from
    console.anthropic.com. A Claude Pro subscription does not include API usage.
    Results from this client are not comparable with the group's (different model).

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

class GPTLLM:
    """GPT via the OpenAI Responses API directly. Not the group standard - use get_llm().

    Requires:
        pip install openai
        OPENAI_API_KEY

    `model` is configurable so experiments can swap GPT models without
    changing agent code.
    """

    def __init__(
        self,
        model: str = "gpt-6-luna",
        reasoning_effort: str = "medium",
        max_tokens: int = 16000,
    ):
        from openai import OpenAI

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens

    def complete(
        self,
        system: str,
        user: str,
        json_schema: Optional[dict] = None,
    ) -> str:

        kwargs = {
            "model": self.model,
            "instructions": system,
            "input": user,
            "max_output_tokens": self.max_tokens,
            "reasoning": {
                "effort": self.reasoning_effort,
            },
        }

        if json_schema is not None:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "agent_output",
                    "schema": json_schema,
                    "strict": True,
                }
            }

        response = self.client.responses.create(
            **kwargs
        )

        if response.status != "completed":
            raise LLMError(
                f"GPT response did not complete: "
                f"status={response.status}, "
                f"details={response.incomplete_details}"
            )

        for item in response.output:
            if getattr(item, "type", None) != "message":
                continue

            for content in item.content:
                if getattr(content, "type", None) == "refusal":
                    raise LLMError(
                        f"GPT refused: {content.refusal}"
                    )

        if not response.output_text:
            raise LLMError(
                "GPT returned no text output"
            )

        return response.output_text


ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _env(name: str) -> Optional[str]:
    """Read a variable from the environment, falling back to the repo's .env file (gitignored)."""
    if os.getenv(name):
        return os.getenv(name)
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, encoding="utf-8") as f:
            for line in f:
                key, sep, value = line.strip().partition("=")
                if sep and key.strip() == name:
                    return value.strip().strip("'\"")
    return None


class OpenRouterLLM:
    """Any model through OpenRouter's OpenAI-compatible API, billed to the group key.

    Requires:
        pip install openai
        OPENROUTER_API_KEY in the environment or in .env (never in code: the repo is on GitHub)

    `model` is an OpenRouter slug, e.g. "anthropic/claude-opus-5.5" or
    "openai/gpt-6.1-sol" (list: https://openrouter.ai/models). No temperature is
    sent, because several reasoning models reject it.
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str = MODEL,
        effort: str = EFFORT,
        max_tokens: int = MAX_TOKENS,
        max_spend_usd: Optional[float] = MAX_SPEND_USD,
    ):
        from openai import OpenAI

        api_key = _env("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMError("OPENROUTER_API_KEY is not set (environment or .env)")
        self.client = OpenAI(base_url=self.BASE_URL, api_key=api_key)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_spend_usd = max_spend_usd
        # Running totals so runs can report what they spent (the group budget is small).
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}

    def complete(self, system: str, user: str, json_schema: Optional[dict] = None) -> str:
        if self.max_spend_usd is not None and self.usage["cost_usd"] >= self.max_spend_usd:
            raise BudgetExceeded(
                f"spend cap reached: ${self.usage['cost_usd']:.2f} of ${self.max_spend_usd:.2f} "
                f"(raise max_spend_usd deliberately if this run really needs more)"
            )
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,  # includes reasoning tokens on most providers
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        extra_body: dict = {"reasoning": {"effort": self.effort, "exclude": True}}
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "agent_output", "strict": True, "schema": json_schema},
            }
            # Only route to providers that actually enforce the schema.
            extra_body["provider"] = {"require_parameters": True}

        response = self.client.chat.completions.create(**kwargs, extra_body=extra_body)

        usage = response.usage
        if usage is not None:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += usage.prompt_tokens or 0
            self.usage["completion_tokens"] += usage.completion_tokens or 0
            self.usage["cost_usd"] += float(getattr(usage, "cost", None) or 0.0)  # OpenRouter's billed cost

        if not response.choices:
            raise LLMError(f"OpenRouter returned no choices: {response}")
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMError("hit max_tokens before finishing (reasoning may have used the budget)")
        if getattr(choice.message, "refusal", None):
            raise LLMError(f"refused: {choice.message.refusal}")
        if not choice.message.content:
            raise LLMError(f"empty response (finish_reason={choice.finish_reason})")
        return choice.message.content


class ScriptedLLM:
    """Offline stand-in for tests: returns queued replies and records every prompt."""

    def __init__(self, replies: list):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, system: str, user: str, json_schema: Optional[dict] = None) -> str:
        self.calls.append({"system": system, "user": user, "json_schema": json_schema})
        reply = self.replies.pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)
