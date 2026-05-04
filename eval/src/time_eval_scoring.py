"""Scoring utilities for time-based market evals."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

INITIAL_CASH = 10_000.0
MAX_CONTRACTS = 100.0
DERIVED_TRADE_EDGE_THRESHOLD = 0.02
MODEL_CUTOFFS: dict[str, pd.Timestamp] = {
    # Verified from public model pages/search snippets on 2026-05-04.
    "openai/gpt-5.5": pd.Timestamp("2025-12-01", tz="UTC"),
    "deepseek/deepseek-v3.2": pd.Timestamp("2025-06-04", tz="UTC"),
}


def parse_probability(text: str) -> float | None:
    patterns = [
        r"P\s*\(\s*YES\s*\)\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(%)?",
        r"PROBABILITY\s+OF\s+YES\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(%)?",
        r"YES\s+PROBABILITY\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(%)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = float(match.group(1))
        is_percent = bool(match.group(2))
        if is_percent:
            if not 0.0 <= value <= 100.0:
                return None
            value /= 100.0
        if not 0.0 <= value <= 1.0:
            return None
        return value
    return None


def derive_trade(prob_yes: float | None, market_price_yes: float) -> dict[str, Any]:
    """Derive a deterministic gross trade from forecast edge vs market price."""
    if prob_yes is None:
        return {"cmd": "INVALID", "side": None, "amount": 0.0, "edge_vs_market": None}

    edge = prob_yes - market_price_yes
    if abs(edge) <= DERIVED_TRADE_EDGE_THRESHOLD:
        return {"cmd": "HOLD", "side": None, "amount": 0.0, "edge_vs_market": edge}

    side = "YES" if edge > 0 else "NO"
    amount = round(min(MAX_CONTRACTS, abs(edge) * MAX_CONTRACTS), 2)
    return {"cmd": "BUY", "side": side, "amount": amount, "edge_vs_market": edge}


def score_forecast(prob_yes: float | None, result: str) -> dict[str, float | None]:
    if prob_yes is None:
        return {"brier": None, "log_loss": None}
    y = 1.0 if result == "yes" else 0.0
    p = min(1.0 - 1e-6, max(1e-6, prob_yes))
    brier = (prob_yes - y) ** 2
    log_loss = -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    return {"brier": brier, "log_loss": log_loss}


def score_trade(action: dict[str, Any], result: str, yes_price: float, no_price: float) -> dict[str, float | str | None]:
    cmd = action["cmd"]
    side = action["side"]
    amount = float(action["amount"])
    if cmd == "HOLD":
        pnl = 0.0
        return {"parsed_action": "HOLD", "contracts": 0.0, "trade_pnl": pnl, "trade_reward": 0.5, "won_trade": None}

    if cmd != "BUY" or side not in {"YES", "NO"} or amount <= 0:
        pnl = 0.0
        return {
            "parsed_action": "INVALID",
            "contracts": 0.0,
            "trade_pnl": pnl,
            "trade_reward": 0.5,
            "won_trade": None,
        }

    price_cents = yes_price if side == "YES" else no_price
    cost = amount * (price_cents / 100.0)
    payout = amount if result == side.lower() else 0.0
    pnl = payout - cost
    reward = (max(-1.0, min(1.0, pnl / INITIAL_CASH)) + 1.0) / 2.0
    return {
        "parsed_action": f"BUY {side}",
        "contracts": amount,
        "trade_pnl": pnl,
        "trade_reward": reward,
        "won_trade": bool(payout > 0),
    }


def _filter_by_model_cutoff(
    rows: list[dict[str, Any]],
    model_name: str | None,
    model_cutoff: pd.Timestamp | str | None,
) -> list[dict[str, Any]]:
    cutoff = pd.Timestamp(model_cutoff) if model_cutoff is not None else None
    if cutoff is None and model_name is not None:
        cutoff = MODEL_CUTOFFS.get(model_name)
    if cutoff is None:
        return rows
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    eligible = []
    for row in rows:
        resolution_date = pd.Timestamp(row["resolution_date"])
        if resolution_date.tzinfo is None:
            resolution_date = resolution_date.tz_localize("UTC")
        else:
            resolution_date = resolution_date.tz_convert("UTC")
        if resolution_date >= cutoff:
            eligible.append(row)
    return eligible


def score_outputs(
    rows: list[dict[str, Any]],
    model_name: str | None = None,
    model_cutoff: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = _filter_by_model_cutoff(rows, model_name=model_name, model_cutoff=model_cutoff)
    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        output_text = str(row.get("output", ""))
        prob_yes = parse_probability(output_text)
        forecast = score_forecast(prob_yes, str(row["result"]))
        baseline_prob_yes = float(row["market_price_at_t0"])
        action = derive_trade(prob_yes, baseline_prob_yes)
        baseline = score_forecast(baseline_prob_yes, str(row["result"]))
        trade = score_trade(
            action,
            str(row["result"]),
            yes_price=float(row["last_yes_price"]),
            no_price=float(row["last_no_price"]),
        )
        scored_rows.append(
            {
                **row,
                "prob_yes": prob_yes,
                "forecast_invalid": prob_yes is None,
                "baseline_prob_yes": baseline_prob_yes,
                "edge_vs_market": action["edge_vs_market"],
                "trade_policy": f"derived_edge_threshold_{DERIVED_TRADE_EDGE_THRESHOLD}",
                **forecast,
                "baseline_brier": baseline["brier"],
                "baseline_log_loss": baseline["log_loss"],
                "brier_delta_vs_baseline": None if forecast["brier"] is None else forecast["brier"] - baseline["brier"],
                "log_loss_delta_vs_baseline": None
                if forecast["log_loss"] is None
                else forecast["log_loss"] - baseline["log_loss"],
                **trade,
                "output": output_text,
            }
        )

    detailed = pd.DataFrame(scored_rows)
    summary = summarize_scores(detailed)
    return detailed, summary


def cluster_bootstrap_ci(
    detailed: pd.DataFrame,
    metric_col: str,
    cluster_col: str = "ticker",
    n_boot: int = 1000,
    seed: int = 7,
    agg: str = "mean",
) -> tuple[float, float]:
    if detailed.empty or metric_col not in detailed or cluster_col not in detailed:
        return (float("nan"), float("nan"))
    cluster_values = []
    for _, group in detailed.groupby(cluster_col):
        values = group[metric_col].dropna()
        if values.empty:
            continue
        if agg == "sum":
            cluster_values.append(float(values.sum()))
        else:
            cluster_values.append(float(values.mean()))
    if not cluster_values:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    estimates = []
    cluster_values_array = np.asarray(cluster_values, dtype=float)
    for _ in range(n_boot):
        sampled = rng.choice(cluster_values_array, size=len(cluster_values_array), replace=True)
        estimates.append(float(np.sum(sampled) if agg == "sum" else np.mean(sampled)))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return (float(low), float(high))


def summarize_scores(detailed: pd.DataFrame) -> pd.DataFrame:
    if detailed.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    groupings: list[tuple[str, str, pd.DataFrame]] = [("all", "all", detailed)]
    groupings.extend(("horizon_bucket", str(name), group) for name, group in detailed.groupby("horizon_bucket"))
    groupings.extend(("category", str(name), group) for name, group in detailed.groupby("category"))
    groupings.extend(
        ("category_x_horizon", f"{category}/{horizon}", group)
        for (category, horizon), group in detailed.groupby(["category", "horizon_bucket"])
    )
    groupings.extend(("liquidity_tier", str(name), group) for name, group in detailed.groupby("liquidity_tier"))
    for group_by, group_name, group in groupings:
        brier_ci_low, brier_ci_high = cluster_bootstrap_ci(group, "brier")
        brier_delta_ci_low, brier_delta_ci_high = cluster_bootstrap_ci(group, "brier_delta_vs_baseline")
        total_pnl_ci_low, total_pnl_ci_high = cluster_bootstrap_ci(group, "trade_pnl", agg="sum")
        rows.append(
            {
                "group_by": group_by,
                "group": group_name,
                "n": len(group),
                "forecast_coverage": group["prob_yes"].notna().mean(),
                "mean_brier": group["brier"].mean(),
                "mean_brier_ci_low": brier_ci_low,
                "mean_brier_ci_high": brier_ci_high,
                "baseline_brier": group["baseline_brier"].mean(),
                "brier_delta_vs_baseline": group["brier_delta_vs_baseline"].mean(),
                "brier_delta_ci_low": brier_delta_ci_low,
                "brier_delta_ci_high": brier_delta_ci_high,
                "mean_log_loss": group["log_loss"].mean(),
                "baseline_log_loss": group["baseline_log_loss"].mean(),
                "log_loss_delta_vs_baseline": group["log_loss_delta_vs_baseline"].mean(),
                "total_pnl": group["trade_pnl"].sum(),
                "total_pnl_ci_low": total_pnl_ci_low,
                "total_pnl_ci_high": total_pnl_ci_high,
                "mean_pnl": group["trade_pnl"].mean(),
                "mean_reward": group["trade_reward"].mean(),
                "trade_rate": group["parsed_action"].isin(["BUY YES", "BUY NO"]).mean(),
                "invalid_rate": (group["parsed_action"] == "INVALID").mean(),
                "forecast_invalid_rate": group["forecast_invalid"].mean(),
                "hold_rate": (group["parsed_action"] == "HOLD").mean(),
                "buy_yes_rate": (group["parsed_action"] == "BUY YES").mean(),
                "buy_no_rate": (group["parsed_action"] == "BUY NO").mean(),
                "win_rate": group["won_trade"].eq(True).mean(),
                "top_abs_pnl_share": _top_abs_pnl_share(group),
            }
        )
    return pd.DataFrame(rows)


def _top_abs_pnl_share(group: pd.DataFrame) -> float:
    abs_pnl = group["trade_pnl"].abs()
    total_abs_pnl = abs_pnl.sum()
    if total_abs_pnl == 0:
        return 0.0
    return float(abs_pnl.max() / total_abs_pnl)
