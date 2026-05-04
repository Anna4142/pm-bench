"""Run OpenRouter models on time-based prediction-market eval tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI

from eval.src.build_time_eval_index import SYSTEM_PROMPT, build_time_eval_index
from eval.src.time_eval_scoring import score_outputs
from eval.src.token_costs import cost_for_usage, count_messages_tokens, fetch_openrouter_pricing, money

ROOT = Path(__file__).resolve().parents[2]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _out_dir_default() -> Path:
    return ROOT / "eval" / "output" / "time_eval"


def _load_or_build_index(
    out_dir: Path,
    markets_dir: Path,
    trades_dir: Path,
    per_bucket: int,
    seed: int,
    rebuild: bool,
) -> pd.DataFrame:
    index_path = out_dir / "time_eval_index.parquet"
    if rebuild or not index_path.exists():
        return build_time_eval_index(
            markets_dir=markets_dir,
            trades_dir=trades_dir,
            out_dir=out_dir,
            per_bucket=per_bucket,
            seed=seed,
        )
    return pd.read_parquet(index_path)


def _sample_tasks(
    index: pd.DataFrame,
    num_examples: int,
    seed: int,
    domains: list[str] | None = None,
) -> pd.DataFrame:
    if domains:
        domain_index = index[index["category"].isin(domains)].copy()
        chunks: list[pd.DataFrame] = []
        for domain in domains:
            for horizon in ["long", "mid", "short"]:
                group = domain_index[(domain_index["category"] == domain) & (domain_index["horizon_bucket"] == horizon)]
                if not group.empty:
                    chunks.append(group.sample(n=1, random_state=seed))
        if not chunks:
            raise SystemExit(f"No tasks matched requested domains: {', '.join(domains)}")
        return pd.concat(chunks, ignore_index=True).sort_values(["category", "horizon_bucket"]).reset_index(drop=True)

    if num_examples <= 0 or num_examples >= len(index):
        return index.copy().reset_index(drop=True)

    per_bucket = max(1, num_examples // max(1, index["horizon_bucket"].nunique()))
    chunks: list[pd.DataFrame] = []
    for _, group in index.groupby("horizon_bucket", sort=False):
        chunks.append(group.sample(n=min(per_bucket, len(group)), random_state=seed))
    sampled = pd.concat(chunks, ignore_index=True)
    remaining = num_examples - len(sampled)
    if remaining > 0:
        rest = index[~index["example_id"].isin(sampled["example_id"])]
        sampled = pd.concat([sampled, rest.sample(n=min(remaining, len(rest)), random_state=seed)], ignore_index=True)
    return sampled.sort_values(["horizon_bucket", "category", "example_id"]).head(num_examples).reset_index(drop=True)


def _messages(row: pd.Series) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(row["prompt"])},
    ]


def _estimate_cost(tasks: pd.DataFrame, model: str, max_tokens: int, api_key: str | None) -> dict[str, Any]:
    pricing = fetch_openrouter_pricing(model, api_key=api_key)
    per_example = []
    for _, row in tasks.iterrows():
        tokens = count_messages_tokens(_messages(row), model)
        per_example.append(
            {
                "example_id": int(row["example_id"]),
                "ticker": row["ticker"],
                "horizon_bucket": row["horizon_bucket"],
                "category": row["category"],
                "liquidity_tier": row["liquidity_tier"],
                "input_tokens": tokens,
            }
        )
    input_tokens = sum(x["input_tokens"] for x in per_example)
    output_tokens = len(tasks) * max_tokens
    return {
        "model": model,
        "num_examples": len(tasks),
        "estimated_input_tokens": input_tokens,
        "max_output_tokens": output_tokens,
        "estimated_max_total_tokens": input_tokens + output_tokens,
        "estimated_max_cost_usd": cost_for_usage(input_tokens, output_tokens, pricing),
        "openrouter_pricing": None
        if pricing is None
        else {
            "prompt_per_million": pricing.prompt_per_million,
            "completion_per_million": pricing.completion_per_million,
            "source": pricing.source,
        },
        "per_example": per_example,
    }


async def _run_one(
    client: AsyncOpenAI,
    model: str,
    row: pd.Series,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=_messages(row),
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"reasoning": {"enabled": False}},
        )
        output = response.choices[0].message.content or ""
        usage = response.usage
        token_usage = {
            "input_tokens": float(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": float(getattr(usage, "completion_tokens", 0) or 0),
        }
        error = None
    except Exception as exc:
        output = ""
        token_usage = {"input_tokens": 0.0, "output_tokens": 0.0}
        error = repr(exc)

    return {
        "example_id": int(row["example_id"]),
        "market_id": row["market_id"],
        "ticker": row["ticker"],
        "question_text": row["question_text"],
        "title": row["title"],
        "resolution_criteria": row["resolution_criteria"],
        "t0": str(row["t0"]),
        "resolution_date": str(row["resolution_date"]),
        "resolution": int(row["resolution"]),
        "horizon_days": float(row["horizon_days"]),
        "horizon_bucket": row["horizon_bucket"],
        "span_bucket": row["span_bucket"],
        "category": row["category"],
        "liquidity_tier": row["liquidity_tier"],
        "result": row["result"],
        "market_price_at_t0": float(row["market_price_at_t0"]),
        "last_yes_price": float(row["last_yes_price"]),
        "last_no_price": float(row["last_no_price"]),
        "lifetime_days": float(row["lifetime_days"]),
        "context_trades": int(row["context_trades"]),
        "output": output,
        "error": error,
        "token_usage": token_usage,
    }


async def run_time_eval(
    model: str,
    num_examples: int,
    max_tokens: int,
    temperature: float,
    max_concurrent: int,
    estimate_only: bool,
    max_estimated_cost: float | None,
    rebuild_index: bool,
    per_bucket: int,
    seed: int,
    domains: list[str] | None,
    run_id: str | None,
    model_cutoff: str | None,
    markets_dir: Path,
    trades_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    _load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")

    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = _safe_name(run_id or started_at)
    index = _load_or_build_index(out_dir, markets_dir, trades_dir, per_bucket, seed, rebuild_index)
    tasks = _sample_tasks(index, num_examples, seed, domains=domains)
    estimate = _estimate_cost(tasks, model=model, max_tokens=max_tokens, api_key=api_key)
    estimate_path = out_dir / f"time_token_estimate__{model.replace('/', '__')}.json"
    estimate_path.write_text(json.dumps(estimate, indent=2))

    print("\nTime-eval token/cost estimate:")
    print(f"  Tasks:                  {len(tasks)}")
    print(f"  Run ID:                 {run_id}")
    print(f"  Temperature:            {temperature}")
    print(f"  Estimated input tokens: {estimate['estimated_input_tokens']:.0f}")
    print(f"  Max output tokens:      {estimate['max_output_tokens']:.0f}")
    print(f"  Expected max cost:      {money(estimate['estimated_max_cost_usd'])}")
    print(f"  Estimate file:          {estimate_path}")

    estimated_cost = estimate["estimated_max_cost_usd"]
    if max_estimated_cost is not None and estimated_cost is not None and estimated_cost > max_estimated_cost:
        raise SystemExit(
            f"Estimated max cost {money(estimated_cost)} exceeds --max-estimated-cost {money(max_estimated_cost)}"
        )
    if estimate_only:
        return {"estimate": estimate}
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set (in shell or in .env)")

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def guarded(row: pd.Series) -> dict[str, Any]:
        async with semaphore:
            return await _run_one(client, model, row, max_tokens=max_tokens, temperature=temperature)

    outputs = await asyncio.gather(*(guarded(row) for _, row in tasks.iterrows()))
    detailed, summary = score_outputs(outputs, model_name=model, model_cutoff=model_cutoff)
    if detailed.empty:
        raise SystemExit("No eligible scored rows remain after model cutoff filtering.")
    detailed.insert(0, "run_id", run_id)
    detailed.insert(1, "model", model)
    detailed.insert(2, "temperature", temperature)

    pricing = fetch_openrouter_pricing(model, api_key=api_key)
    actual_input = sum(o["token_usage"]["input_tokens"] for o in outputs)
    actual_output = sum(o["token_usage"]["output_tokens"] for o in outputs)
    actual_cost = cost_for_usage(actual_input, actual_output, pricing)

    safe_model = model.replace("/", "__")
    run_suffix = f"{safe_model}__{run_id}"
    results_path = out_dir / f"time_model_results__{safe_model}.jsonl"
    detailed_path = out_dir / f"time_model_scored__{safe_model}.csv"
    summary_path = out_dir / f"time_model_summary__{safe_model}.csv"
    metadata_path = out_dir / f"time_model_metadata__{safe_model}.json"
    run_results_path = out_dir / f"time_model_results__{run_suffix}.jsonl"
    run_detailed_path = out_dir / f"time_model_scored__{run_suffix}.csv"
    run_task_log_path = out_dir / f"time_model_task_log__{run_suffix}.csv"
    run_invalid_path = out_dir / f"time_model_invalid_outputs__{run_suffix}.csv"
    run_summary_path = out_dir / f"time_model_summary__{run_suffix}.csv"
    run_metadata_path = out_dir / f"time_model_metadata__{run_suffix}.json"

    with run_results_path.open("w") as f:
        for output in outputs:
            f.write(json.dumps(output, default=str) + "\n")
    results_path.write_text(run_results_path.read_text())
    detailed.to_csv(detailed_path, index=False)
    detailed.to_csv(run_detailed_path, index=False)
    task_log_columns = [
        "run_id",
        "model",
        "temperature",
        "example_id",
        "market_id",
        "ticker",
        "category",
        "horizon_bucket",
        "liquidity_tier",
        "t0",
        "resolution_date",
        "market_price_at_t0",
        "resolution",
        "result",
        "prob_yes",
        "forecast_invalid",
        "edge_vs_market",
        "trade_policy",
        "parsed_action",
        "contracts",
        "trade_pnl",
        "brier",
        "log_loss",
        "baseline_brier",
        "brier_delta_vs_baseline",
        "output",
    ]
    detailed[[col for col in task_log_columns if col in detailed.columns]].to_csv(run_task_log_path, index=False)
    invalid_outputs = detailed[detailed["forecast_invalid"] | (detailed["parsed_action"] == "INVALID")]
    invalid_outputs.to_csv(run_invalid_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(run_summary_path, index=False)
    metadata = {
        "run_id": run_id,
        "model": model,
        "temperature": temperature,
        "started_at": started_at,
        "domains": domains,
        "seed": seed,
        "model_cutoff": model_cutoff,
        "n_requested": len(outputs),
        "n_eligible": len(detailed),
        "estimate": estimate,
        "actual_usage": {
            "actual_input_tokens": actual_input,
            "actual_output_tokens": actual_output,
            "actual_total_tokens": actual_input + actual_output,
            "actual_cost_usd": actual_cost,
        },
        "summary": summary.to_dict("records"),
        "paths": {
            "results": str(results_path),
            "detailed": str(detailed_path),
            "summary": str(summary_path),
            "run_results": str(run_results_path),
            "run_detailed": str(run_detailed_path),
            "run_task_log": str(run_task_log_path),
            "run_invalid_outputs": str(run_invalid_path),
            "run_summary": str(run_summary_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    run_metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    print("\nTime-eval summary:")
    print(summary.to_string(index=False))
    print(f"\nActual token usage: {actual_input:.0f} input, {actual_output:.0f} output, {money(actual_cost)}")
    print(f"Raw outputs: {results_path}")
    print(f"Scored rows: {detailed_path}")
    print(f"Run task log: {run_task_log_path}")
    print(f"Invalid output audit: {run_invalid_path}")
    print(f"Summary: {summary_path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-examples", type=int, default=9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--max-estimated-cost", type=float, default=None)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--domains", nargs="*", default=None, help="Optional categories to cross with long/mid/short.")
    parser.add_argument("--run-id", default=None, help="Optional ID for non-overwritten run artifacts.")
    parser.add_argument("--model-cutoff", default=None, help="Exclude tasks with resolution_date before this date.")
    parser.add_argument("--per-bucket", type=int, default=int(os.environ.get("TIME_EVAL_PER_BUCKET", "20")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("EVAL_SEED", "7")))
    parser.add_argument("--markets-dir", type=Path, default=ROOT / "data" / "kalshi" / "markets")
    parser.add_argument("--trades-dir", type=Path, default=ROOT / "data" / "kalshi" / "trades")
    parser.add_argument("--out-dir", type=Path, default=_out_dir_default())
    args = parser.parse_args()

    asyncio.run(
        run_time_eval(
            model=args.model,
            num_examples=args.num_examples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_concurrent=args.max_concurrent,
            estimate_only=args.estimate_only,
            max_estimated_cost=args.max_estimated_cost,
            rebuild_index=args.rebuild_index,
            per_bucket=args.per_bucket,
            seed=args.seed,
            domains=args.domains,
            run_id=args.run_id,
            model_cutoff=args.model_cutoff,
            markets_dir=args.markets_dir,
            trades_dir=args.trades_dir,
            out_dir=args.out_dir,
        )
    )


if __name__ == "__main__":
    main()
