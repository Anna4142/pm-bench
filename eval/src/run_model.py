"""Run the prediction-market trading env via OpenRouter.

Reads ``OPENROUTER_API_KEY`` from environment (or ``prediction-market-analysis/.env``).
Loads the env from the local package, evaluates ``num_examples`` rollouts, and
appends a ``model`` row to ``baseline_summary.csv`` for the existing report.

Usage::

    uv run python -m eval.src.run_model --model openai/gpt-4o --num-examples 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

from eval.src.token_costs import (
    aggregate_actual_usage,
    estimate_run_cost,
    fetch_openrouter_pricing,
    money,
)

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _ensure_imports() -> None:
    for path in (
        ROOT / "environments" / "prediction_market_trading",
        ROOT.parent / "verifiers",
    ):
        if path.exists():
            sys.path.insert(0, str(path))


def _markets_dir_default() -> str:
    return str(ROOT / "data" / "kalshi" / "markets")


async def run_eval(
    model: str,
    num_examples: int,
    markets_dir: str | None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    estimate_only: bool = False,
    max_estimated_cost: float | None = None,
) -> dict:
    from openai import AsyncOpenAI
    from prediction_market_trading import load_environment  # type: ignore[import-not-found]

    api_key = os.environ.get("OPENROUTER_API_KEY")

    env = load_environment(
        markets_dir=markets_dir,
        num_train_examples=max(num_examples, 5),
        num_eval_examples=num_examples,
    )

    out_dir = ROOT / "eval" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"model_results__{model.replace('/', '__')}.json"
    estimate_path = out_dir / f"token_estimate__{model.replace('/', '__')}.json"

    pricing = fetch_openrouter_pricing(model, api_key=api_key)
    token_estimate = estimate_run_cost(
        env=env,
        model=model,
        num_examples=num_examples,
        max_tokens=max_tokens,
        pricing=pricing,
    )
    estimate_path.write_text(json.dumps(token_estimate, indent=2))

    print("\nToken/cost estimate:")
    print(f"  Estimated input tokens: {token_estimate['estimated_input_tokens']:.0f}")
    print(f"  Max output tokens:       {token_estimate['max_output_tokens']:.0f}")
    print(f"  Max total tokens:        {token_estimate['estimated_max_total_tokens']:.0f}")
    print(f"  Expected max cost:       {money(token_estimate['estimated_max_cost_usd'])}")
    print(f"  Estimate file:           {estimate_path}")

    estimated_cost = token_estimate["estimated_max_cost_usd"]
    if max_estimated_cost is not None and estimated_cost is not None and estimated_cost > max_estimated_cost:
        raise SystemExit(
            f"Estimated max cost {money(estimated_cost)} exceeds --max-estimated-cost {money(max_estimated_cost)}"
        )

    if estimate_only:
        return token_estimate

    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set (in shell or in .env)")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    if raw_path.exists():
        if raw_path.is_dir():
            shutil.rmtree(raw_path)
        else:
            raw_path.unlink()

    # Verifiers renames max_tokens -> max_completion_tokens for chat. Some
    # OpenRouter backends still allocate large default reasoning budgets unless
    # capped; keep both keys + a small extra_body hint for reasoning models.
    sampling_args: dict = {
        "max_tokens": max_tokens,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "extra_body": {
            "reasoning": {"enabled": False},
        },
    }
    print(f"Evaluating {model} on {num_examples} examples (cap={max_tokens} completion tokens) ...")
    results = await env.evaluate(
        client=client,
        model=model,
        sampling_args=sampling_args,
        num_examples=num_examples,
        rollouts_per_example=1,
        max_concurrent=1,
        save_results=True,
        results_path=raw_path,
    )

    outputs = results.get("outputs") or []
    rewards = [float(o.get("reward", 0.0)) for o in outputs]

    pnls: list[float] = []
    for o in outputs:
        pnl = o.get("final_pnl")
        if pnl is None and isinstance(o.get("metrics"), dict):
            # proxy when rollout errored before final_pnl was flattened
            pnl = o["metrics"].get("cash_metric", 0.0) - 10000.0
        try:
            pnls.append(float(pnl) if pnl is not None else 0.0)
        except (TypeError, ValueError):
            pnls.append(0.0)

    n = len(outputs)
    err_n = sum(1 for o in outputs if o.get("error"))
    actual_usage = aggregate_actual_usage(outputs, pricing=pricing)
    summary = {
        "policy": f"model:{model}",
        "n": n,
        "errors": err_n,
        "mean_pnl": (sum(pnls) / n) if n else 0.0,
        "total_pnl": sum(pnls) if pnls else 0.0,
        "mean_reward": (sum(rewards) / n) if n else 0.0,
        "win_rate": (sum(1 for p in pnls if p > 0) / n) if n and pnls else 0.0,
        "estimated_input_tokens": token_estimate["estimated_input_tokens"],
        "max_output_tokens": token_estimate["max_output_tokens"],
        "estimated_max_cost_usd": token_estimate["estimated_max_cost_usd"],
        "actual_input_tokens": actual_usage["actual_input_tokens"],
        "actual_output_tokens": actual_usage["actual_output_tokens"],
        "actual_total_tokens": actual_usage["actual_total_tokens"],
        "actual_cost_usd": actual_usage["actual_cost_usd"],
    }

    summary_csv = out_dir / "baseline_summary.csv"
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        df = df[df["policy"] != summary["policy"]]
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    else:
        df = pd.DataFrame([summary])
    df.to_csv(summary_csv, index=False)

    summary_json_path = out_dir / "model_summary.json"
    summary_json_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "token_estimate": token_estimate,
                "actual_usage": actual_usage,
            },
            indent=2,
        )
    )

    print("\nModel summary:")
    print(json.dumps(summary, indent=2))
    print(
        "\nActual token usage: "
        f"{actual_usage['actual_input_tokens']:.0f} input, "
        f"{actual_usage['actual_output_tokens']:.0f} output, "
        f"{money(actual_usage['actual_cost_usd'])}"
    )
    print(f"\nRaw rollouts: {raw_path}")
    print(f"Updated summary: {summary_csv}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="OpenRouter model id, e.g. openai/gpt-4o")
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print/write token and OpenRouter cost estimates without making model calls.",
    )
    parser.add_argument(
        "--max-estimated-cost",
        type=float,
        default=None,
        help="Abort before running if the estimated max OpenRouter cost in USD is higher than this value.",
    )
    parser.add_argument(
        "--markets-dir",
        default=os.environ.get("EVAL_MARKETS_DIR", _markets_dir_default()),
    )
    args = parser.parse_args()

    _load_dotenv()
    _ensure_imports()
    asyncio.run(
        run_eval(
            args.model,
            args.num_examples,
            args.markets_dir,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            estimate_only=args.estimate_only,
            max_estimated_cost=args.max_estimated_cost,
        )
    )


if __name__ == "__main__":
    main()
