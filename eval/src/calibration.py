"""Build a price -> empirical win rate calibration table from the train pool.

For each integer ``yes_ask`` price (1..99 cents) we estimate
``P(result == 'yes' | yes_ask == p)`` using the *train* split only. The eval
slice never contributes, so any policy built from this table is leakage-free.

Writes:
    {out_dir}/calibration_curve.csv  # columns: price, win_rate, n
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def build_calibration(train_df: pd.DataFrame) -> pd.DataFrame:
    """Empirical YES-resolves rate per yes_ask price."""
    if train_df.empty:
        return pd.DataFrame(columns=["price", "win_rate", "n"])

    g = (
        train_df.assign(_yes=(train_df["result"] == "yes").astype(int))
        .groupby("yes_ask")
        .agg(win_rate=("_yes", "mean"), n=("_yes", "size"))
        .reset_index()
        .rename(columns={"yes_ask": "price"})
        .sort_values("price")
    )

    # Smooth tiny buckets toward the midpoint of the price (so we don't get
    # 100% / 0% from <5 samples). Simple Laplace-style shrinkage.
    K = 5  # pseudo-count strength
    prior = g["price"] / 100.0
    g["win_rate"] = (g["win_rate"] * g["n"] + prior * K) / (g["n"] + K)
    return g.reset_index(drop=True)


def main(out_dir: Path) -> Path:
    train_path = out_dir / "train_pool.parquet"
    if not train_path.exists():
        raise SystemExit(f"Missing {train_path}; run build_eval_index first")
    train_df = pd.read_parquet(train_path)
    curve = build_calibration(train_df)
    out = out_dir / "calibration_curve.csv"
    curve.to_csv(out, index=False)
    print(f"calibration: wrote {len(curve)} price rows -> {out}")
    return out


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[2]
    main(Path(os.environ.get("EVAL_OUT_DIR", here / "eval" / "output")))
