"""Scoring utilities for time-based market evals."""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

INITIAL_CASH = 10_000.0
MAX_CONTRACTS = 100.0
DERIVED_TRADE_EDGE_THRESHOLD = 0.02


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


def parse_action(text: str, ticker: str) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    ticker_upper = ticker.upper()
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("`*_ ")
        line = re.sub(r"^[\s>*#\-\d\.\)]+", "", line).strip()
        line = re.sub(
            r"^(?:(?:FINAL|TRADE)\s+)?(?:DECISION|ACTION|TRADE)\s*:\s*",
            "",
            line,
            flags=re.IGNORECASE,
        )
        upper = line.upper()
        if re.match(r"^(?:MY\s+DECISION\s+IS\s+)?HOLD\b", upper):
            candidates.append({"cmd": "HOLD", "side": None, "amount": 0.0})
            continue

        match = re.match(
            r"^(?:MY\s+DECISION\s+IS\s+)?(BUY|SELL)\s+(YES|NO)\b(?:\s+([A-Z0-9._:-]+))?(?:\s+([\d,]+(?:\.\d+)?))?",
            upper,
        )
        if not match:
            continue
        cmd, side, parsed_ticker, amount_text = match.groups()
        if parsed_ticker and parsed_ticker not in {ticker_upper, "TICKER", "MARKET", "MARKET_ID"}:
            continue
        amount = float(amount_text.replace(",", "")) if amount_text and amount_text != "CONTRACTS" else MAX_CONTRACTS
        candidates.append({"cmd": cmd, "side": side, "amount": min(MAX_CONTRACTS, max(0.0, amount))})

    if candidates:
        return candidates[-1]
    return {"cmd": "INVALID", "side": None, "amount": 0.0}


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


def score_outputs(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        rows.append(
            {
                "group_by": group_by,
                "group": group_name,
                "n": len(group),
                "forecast_coverage": group["prob_yes"].notna().mean(),
                "mean_brier": group["brier"].mean(),
                "baseline_brier": group["baseline_brier"].mean(),
                "brier_delta_vs_baseline": group["brier_delta_vs_baseline"].mean(),
                "mean_log_loss": group["log_loss"].mean(),
                "baseline_log_loss": group["baseline_log_loss"].mean(),
                "log_loss_delta_vs_baseline": group["log_loss_delta_vs_baseline"].mean(),
                "total_pnl": group["trade_pnl"].sum(),
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
