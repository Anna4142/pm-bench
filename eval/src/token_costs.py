"""Token and OpenRouter cost accounting for eval runs."""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


@dataclass(frozen=True)
class OpenRouterPricing:
    prompt_per_token: float
    completion_per_token: float
    source: str

    @property
    def prompt_per_million(self) -> float:
        return self.prompt_per_token * 1_000_000

    @property
    def completion_per_million(self) -> float:
        return self.completion_per_token * 1_000_000


def _fallback_token_count(text: str) -> int:
    """Conservative-ish fallback when tiktoken is unavailable."""
    if not text:
        return 0
    word_pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    char_estimate = math.ceil(len(text) / 4)
    return max(1, max(char_estimate, len(word_pieces)))


def _token_count(text: str, model: str) -> int:
    try:
        import tiktoken  # type: ignore[import-not-found]

        try:
            encoding = tiktoken.encoding_for_model(model)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return _fallback_token_count(text)


def count_messages_tokens(messages: list[dict[str, Any]], model: str) -> int:
    # ChatML-style overhead estimate. Actual OpenRouter provider tokenizers vary,
    # so post-run accounting should use provider-reported token_usage.
    total = 2
    for msg in messages:
        total += 4
        total += _token_count(str(msg.get("role", "")), model)
        total += _token_count(str(msg.get("content", "")), model)
    return total


def estimate_prompt_tokens(env: Any, model: str, num_examples: int) -> dict[str, Any]:
    eval_dataset = getattr(env, "eval_dataset", None)
    system_prompt = getattr(env, "system_prompt", "")
    per_example: list[dict[str, Any]] = []

    if eval_dataset is None:
        return {
            "tokenizer": "tiktoken_or_fallback",
            "input_tokens": 0,
            "per_example": per_example,
            "warning": "env.eval_dataset was not available",
        }

    for i in range(min(num_examples, len(eval_dataset))):
        row = eval_dataset[i]
        messages = [{"role": "system", "content": system_prompt}, *list(row.get("prompt", []))]
        tokens = count_messages_tokens(messages, model)
        info = row.get("info") or {}
        per_example.append(
            {
                "example_id": i,
                "ticker": info.get("ticker") or row.get("ticker"),
                "input_tokens": tokens,
            }
        )

    return {
        "tokenizer": "tiktoken_or_fallback",
        "input_tokens": sum(x["input_tokens"] for x in per_example),
        "per_example": per_example,
    }


def fetch_openrouter_pricing(model: str, api_key: str | None = None, timeout_seconds: float = 10.0) -> OpenRouterPricing | None:
    request = urllib.request.Request(OPENROUTER_MODELS_URL)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    for item in payload.get("data", []):
        if item.get("id") != model:
            continue
        pricing = item.get("pricing") or {}
        try:
            return OpenRouterPricing(
                prompt_per_token=float(pricing.get("prompt", 0.0)),
                completion_per_token=float(pricing.get("completion", 0.0)),
                source=OPENROUTER_MODELS_URL,
            )
        except (TypeError, ValueError):
            return None

    return None


def cost_for_usage(input_tokens: float, output_tokens: float, pricing: OpenRouterPricing | None) -> float | None:
    if pricing is None:
        return None
    return (input_tokens * pricing.prompt_per_token) + (output_tokens * pricing.completion_per_token)


def estimate_run_cost(
    env: Any,
    model: str,
    num_examples: int,
    max_tokens: int,
    pricing: OpenRouterPricing | None,
) -> dict[str, Any]:
    prompt_estimate = estimate_prompt_tokens(env, model, num_examples)
    input_tokens = int(prompt_estimate["input_tokens"])
    max_output_tokens = num_examples * max_tokens
    estimated_cost = cost_for_usage(input_tokens, max_output_tokens, pricing)

    return {
        "model": model,
        "num_examples": num_examples,
        "estimated_input_tokens": input_tokens,
        "max_output_tokens": max_output_tokens,
        "estimated_max_total_tokens": input_tokens + max_output_tokens,
        "openrouter_pricing": None
        if pricing is None
        else {
            "prompt_per_million": pricing.prompt_per_million,
            "completion_per_million": pricing.completion_per_million,
            "source": pricing.source,
        },
        "estimated_max_cost_usd": estimated_cost,
        "per_example": prompt_estimate["per_example"],
    }


def aggregate_actual_usage(outputs: list[dict[str, Any]], pricing: OpenRouterPricing | None) -> dict[str, Any]:
    input_tokens = 0.0
    output_tokens = 0.0

    for output in outputs:
        usage = output.get("token_usage") or {}
        input_tokens += float(usage.get("input_tokens") or usage.get("prompt_tokens") or 0.0)
        output_tokens += float(usage.get("output_tokens") or usage.get("completion_tokens") or 0.0)

    actual_cost = cost_for_usage(input_tokens, output_tokens, pricing)
    return {
        "actual_input_tokens": input_tokens,
        "actual_output_tokens": output_tokens,
        "actual_total_tokens": input_tokens + output_tokens,
        "actual_cost_usd": actual_cost,
    }


def money(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value < 0.01:
        return f"${value:.6f}"
    return f"${value:.4f}"
