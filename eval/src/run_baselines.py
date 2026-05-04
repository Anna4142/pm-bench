"""Run baseline policies on the eval index and persist per-row PnL + summary."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from eval.src.baselines import POLICIES, episode_pnl


def _load_calibration(out_dir: Path) -> dict[int, float]:
    csv = out_dir / "calibration_curve.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv)
    return {int(r.price): float(r.win_rate) for r in df.itertuples()}


def main(out_dir: Path) -> dict[str, Path]:
    eval_path = out_dir / "eval_index.parquet"
    if not eval_path.exists():
        raise SystemExit(f"Missing {eval_path}; run build_eval_index first")
    eval_df = pd.read_parquet(eval_path)
    calib = _load_calibration(out_dir)

    rows: list[dict] = []
    for policy_name, policy in POLICIES.items():
        for _, market in eval_df.iterrows():
            action = policy(market, calib)
            pnl = episode_pnl(action, market)
            side = action[0] if action else "HOLD"
            rows.append(
                {
                    "policy": policy_name,
                    "ticker": market["ticker"],
                    "bucket": market["bucket"],
                    "yes_ask": int(market["yes_ask"]),
                    "no_ask": int(market["no_ask"]),
                    "result": market["result"],
                    "action_side": side,
                    "pnl": pnl,
                }
            )

    results = pd.DataFrame(rows)
    results_path = out_dir / "baseline_results.parquet"
    results.to_parquet(results_path, index=False)

    summary = (
        results.groupby("policy")
        .agg(
            n=("pnl", "size"),
            mean_pnl=("pnl", "mean"),
            total_pnl=("pnl", "sum"),
            win_rate=("pnl", lambda x: float((x > 0).mean())),
            traded_rate=("action_side", lambda s: float((s != "HOLD").mean())),
        )
        .reset_index()
        .sort_values("mean_pnl", ascending=False)
    )

    bucket_summary = (
        results.groupby(["policy", "bucket"])
        .agg(n=("pnl", "size"), mean_pnl=("pnl", "mean"))
        .reset_index()
        .sort_values(["policy", "bucket"])
    )

    summary_path = out_dir / "baseline_summary.csv"
    bucket_path = out_dir / "baseline_summary_by_bucket.csv"
    summary.to_csv(summary_path, index=False)
    bucket_summary.to_csv(bucket_path, index=False)

    (out_dir / "baseline_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2)
    )

    print("\nbaseline summary (PnL is $ per contract per market):")
    print(summary.to_string(index=False))

    return {
        "results": results_path,
        "summary": summary_path,
        "by_bucket": bucket_path,
    }


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[2]
    main(Path(os.environ.get("EVAL_OUT_DIR", here / "eval" / "output")))
