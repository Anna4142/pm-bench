"""Re-score time_eval results under alternative trade-sizing rules.

Reads a `time_model_scored__*.csv` produced by `run_time_model.py` and
re-computes the simulated PnL under several deterministic trade-sizing
policies, all using the same forecast `prob_yes` and the same execution
prices (`last_yes_price`, `last_no_price`).

The forecast quality (Brier, log-loss, calibration) is unchanged across
policies; only `contracts` and `trade_pnl` differ.

Usage:
    python -m eval.src.rescore_trades \\
        --scored eval/output/time_eval/time_model_scored__deepseek__deepseek-v3.2.csv \\
        --out    eval/output/time_eval/time_model_rescored__deepseek__deepseek-v3.2.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

INITIAL_CASH = 10_000.0
MAX_CONTRACTS = 100.0
DERIVED_TRADE_EDGE_THRESHOLD = 0.05
N_BOOT = 1000
BOOT_SEED = 7


# ---------------------------------------------------------------------------
# Sizing rules. Each rule returns `contracts` given:
#   edge          = prob_yes - decision_price
#   decision_price= the price the model compared against (market VWAP at t0)
#   exec_price    = the price the trade actually clears at (last tick before t0)
#   side          = "YES" or "NO"
#
# CRITICAL: dollar caps must be applied using exec_price, because that is the
# price the position is taken at. For a BUY at exec price q with n contracts,
# the upfront cost = n*q and (since this is binary cash-or-nothing) the
# worst-case loss is exactly n*q. So a "max-loss-per-trade = $X" cap means
# n*q <= X, i.e. n <= X/q.
# ---------------------------------------------------------------------------


def size_fixed_edge(edge: float, decision_price: float, exec_price: float, side: str) -> float:
    """Original rule: contracts = min(100, |edge| * 100). Used in the headline run."""
    del decision_price, exec_price, side
    return float(min(MAX_CONTRACTS, abs(edge) * MAX_CONTRACTS))


def size_kelly(
    edge: float,
    decision_price: float,
    exec_price: float,
    side: str,
    kelly_fraction: float = 0.25,
    max_dollars: float = 25.0,
) -> float:
    """Fractional-Kelly sizing with a hard per-trade dollar cap (applied at exec price).

    Kelly fraction for BUY YES at decision price q with belief p: (p - q) / (1 - q).
    Kelly fraction for BUY NO  at decision price q with belief p: (q - p) / q.

    Target dollars = min(max_dollars, kelly_fraction * f * INITIAL_CASH).
    Contracts = target_dollars / exec_cost_per_contract.

    The cap is binding on actual upfront cost, so worst-case loss <= max_dollars.
    """
    p = decision_price + edge
    q = decision_price
    if side == "YES":
        f = max(0.0, (p - q) / max(1e-6, 1.0 - q))
    else:
        f = max(0.0, (q - p) / max(1e-6, q))
    if f <= 0.0:
        return 0.0
    target_dollars = min(max_dollars, kelly_fraction * f * INITIAL_CASH)
    contracts = target_dollars / max(1e-6, exec_price)
    return float(min(MAX_CONTRACTS, contracts))


def size_loss_capped(
    edge: float,
    decision_price: float,
    exec_price: float,
    side: str,
    max_loss_dollars: float = 5.0,
) -> float:
    """Edge-prop sizing capped so that the per-trade dollar loss <= max_loss_dollars.

    base contracts = |edge| * 100 (capped at 100), then n <= max_loss / exec_cost_per_contract.
    """
    del decision_price, side
    base = min(MAX_CONTRACTS, abs(edge) * MAX_CONTRACTS)
    cap = max_loss_dollars / max(1e-6, exec_price)
    return float(min(base, cap))


SIZING_RULES = {
    "fixed_edge_100": dict(fn=size_fixed_edge, label="size = |edge| * 100, max 100 (original)"),
    "kelly_q4_cap25": dict(
        fn=lambda e, q, x, s: size_kelly(e, q, x, s, kelly_fraction=0.25, max_dollars=25.0),
        label="quarter-Kelly, $25/trade cap (exec-priced)",
    ),
    "kelly_q4_cap10": dict(
        fn=lambda e, q, x, s: size_kelly(e, q, x, s, kelly_fraction=0.25, max_dollars=10.0),
        label="quarter-Kelly, $10/trade cap (exec-priced)",
    ),
    "loss_cap_5": dict(
        fn=lambda e, q, x, s: size_loss_capped(e, q, x, s, max_loss_dollars=5.0),
        label="edge-prop, $5 max-loss/trade (exec-priced)",
    ),
    "loss_cap_2": dict(
        fn=lambda e, q, x, s: size_loss_capped(e, q, x, s, max_loss_dollars=2.0),
        label="edge-prop, $2 max-loss/trade (exec-priced)",
    ),
}


# ---------------------------------------------------------------------------
# PnL given size + execution prices.
# ---------------------------------------------------------------------------


def compute_pnl(side: str, contracts: float, exec_price_cents: float, result_yes: int) -> float:
    """PnL for a BUY at the given execution price (cents 0-100).

    Returns dollar PnL = payout - cost.
    For BUY YES: cost = contracts * exec_price/100, payout = contracts iff result_yes.
    For BUY NO : cost = contracts * exec_price/100, payout = contracts iff !result_yes.
    """
    if contracts <= 0.0:
        return 0.0
    cost = contracts * (exec_price_cents / 100.0)
    win = (side == "YES" and result_yes == 1) or (side == "NO" and result_yes == 0)
    payout = contracts if win else 0.0
    return float(payout - cost)


# ---------------------------------------------------------------------------
# Cluster bootstrap on (cluster_col -> mean PnL across cluster).
# ---------------------------------------------------------------------------


def cluster_mean_ci(df: pd.DataFrame, value_col: str, cluster_col: str = "ticker") -> tuple[float, float]:
    if df.empty or value_col not in df or cluster_col not in df:
        return (float("nan"), float("nan"))
    cluster_means: list[float] = []
    for _, group in df.groupby(cluster_col):
        v = group[value_col].dropna()
        if not v.empty:
            cluster_means.append(float(v.mean()))
    if not cluster_means:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(BOOT_SEED)
    arr = np.asarray(cluster_means)
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(N_BOOT)]
    low, high = np.percentile(boots, [2.5, 97.5])
    return (float(low), float(high))


# ---------------------------------------------------------------------------
# Main rescoring pipeline.
# ---------------------------------------------------------------------------


def rescore(scored: pd.DataFrame) -> pd.DataFrame:
    df = scored.copy()
    result_yes = df["result"].astype(str).str.lower().eq("yes").astype(int)

    for rule_name, rule in SIZING_RULES.items():
        fn = rule["fn"]
        contracts_col = []
        pnl_col = []
        for _, row in df.iterrows():
            edge = row.get("edge_vs_market")
            prob_yes = row.get("prob_yes")
            decision_price = row.get("market_price_at_t0")
            if pd.isna(edge) or pd.isna(prob_yes) or pd.isna(decision_price):
                contracts_col.append(0.0)
                pnl_col.append(0.0)
                continue
            edge = float(edge)
            decision_price = float(decision_price)
            if abs(edge) <= DERIVED_TRADE_EDGE_THRESHOLD:
                contracts_col.append(0.0)
                pnl_col.append(0.0)
                continue
            side = "YES" if edge > 0 else "NO"
            exec_price_cents = float(row["last_yes_price"]) if side == "YES" else float(row["last_no_price"])
            exec_price = exec_price_cents / 100.0
            n = fn(edge, decision_price, exec_price, side)
            pnl = compute_pnl(side, n, exec_price_cents, int(result_yes.loc[row.name]))
            contracts_col.append(round(n, 4))
            pnl_col.append(round(pnl, 4))
        df[f"contracts__{rule_name}"] = contracts_col
        df[f"trade_pnl__{rule_name}"] = pnl_col

    return df


def summarize(rescored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for rule_name, rule in SIZING_RULES.items():
        col = f"trade_pnl__{rule_name}"
        contracts_col = f"contracts__{rule_name}"
        traded = rescored[rescored[contracts_col] > 0]
        n_traded = int(len(traded))
        n_total = int(len(rescored))
        total_pnl = float(rescored[col].sum())
        mean_pnl = float(rescored[col].mean())
        mean_pnl_lo, mean_pnl_hi = cluster_mean_ci(rescored, col, cluster_col="ticker")
        win_rate = float((traded[col] > 0).mean()) if n_traded else float("nan")
        max_loss = float(traded[col].min()) if n_traded else 0.0
        max_win = float(traded[col].max()) if n_traded else 0.0
        abs_top1 = float(traded[col].abs().max() / max(1e-6, traded[col].abs().sum())) if n_traded else float("nan")
        rows.append(
            {
                "rule": rule_name,
                "label": rule["label"],
                "n_traded": n_traded,
                "trade_rate": round(n_traded / max(1, n_total), 4),
                "total_pnl": round(total_pnl, 2),
                "mean_pnl": round(mean_pnl, 4),
                "mean_pnl_ci_low": round(mean_pnl_lo, 4),
                "mean_pnl_ci_high": round(mean_pnl_hi, 4),
                "win_rate": round(win_rate, 4),
                "max_loss": round(max_loss, 2),
                "max_win": round(max_win, 2),
                "abs_pnl_top1_share": round(abs_top1, 4),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()

    scored = pd.read_csv(args.scored)
    print(f"Loaded {len(scored)} rows from {args.scored}")

    rescored = rescore(scored)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rescored.to_csv(args.out, index=False)
    print(f"Wrote rescored CSV to {args.out}")

    summary = summarize(rescored)
    summary_out = args.summary_out or args.out.with_name(args.out.stem + "__summary.csv")
    summary.to_csv(summary_out, index=False)
    print(f"Wrote summary CSV to {summary_out}\n")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
