"""Filter finalized Kalshi markets and stratify-sample them into an eval index.

Reads parquet under ``markets_dir``, keeps rows that:
  - status == 'finalized'
  - result in ('yes', 'no')
  - 1 <= yes_ask <= 99 and 1 <= no_ask <= 99
  - volume_24h > 0

Splits into train / eval halves by ``created_time`` (the eval half is later in
time so calibration learned on the train half doesn't leak), then stratifies
the eval half by ``yes_ask`` price bucket and samples ``per_bucket`` markets
per bucket.

Writes:
    {out_dir}/markets_filtered.parquet      # full filtered corpus + bucket col
    {out_dir}/eval_index.parquet            # the curated eval slice
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

PRICE_BUCKETS: list[tuple[int, int]] = [
    (1, 10),
    (11, 25),
    (26, 49),
    (50, 50),
    (51, 74),
    (75, 90),
    (91, 99),
]


def _bucket_label(p: int) -> str:
    for lo, hi in PRICE_BUCKETS:
        if lo <= p <= hi:
            return f"{lo:02d}-{hi:02d}"
    return "other"


def load_filtered_markets(markets_dir: Path) -> pd.DataFrame:
    """Load finalized, sane-priced markets from parquet."""
    glob = str(markets_dir / "*.parquet")
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT
            ticker,
            title,
            yes_bid, yes_ask,
            no_bid,  no_ask,
            volume_24h,
            close_time,
            created_time,
            result
        FROM '{glob}'
        WHERE status = 'finalized'
          AND result IN ('yes', 'no')
          AND yes_ask BETWEEN 1 AND 99
          AND no_ask  BETWEEN 1 AND 99
          AND volume_24h > 0
        """
    ).df()
    con.close()
    df["yes_ask"] = df["yes_ask"].astype(int)
    df["no_ask"] = df["no_ask"].astype(int)
    df["bucket"] = df["yes_ask"].map(_bucket_label)
    df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce")
    return df.sort_values("created_time").reset_index(drop=True)


def split_train_eval(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split: earlier half = train (for calibration), later half = eval pool."""
    if df.empty:
        return df, df
    cut = len(df) // 2
    train = df.iloc[:cut].copy()
    eval_pool = df.iloc[cut:].copy()
    return train, eval_pool


def stratified_sample(
    eval_pool: pd.DataFrame,
    per_bucket: int,
    seed: int,
) -> pd.DataFrame:
    """Sample up to ``per_bucket`` rows per price bucket."""
    if eval_pool.empty:
        return eval_pool
    chunks: list[pd.DataFrame] = []
    for bucket, grp in eval_pool.groupby("bucket"):
        n = min(len(grp), per_bucket)
        chunks.append(grp.sample(n=n, random_state=seed))
    return (
        pd.concat(chunks, axis=0)
        .sort_values(["bucket", "created_time"])
        .reset_index(drop=True)
    )


def main(
    markets_dir: Path,
    out_dir: Path,
    per_bucket: int,
    seed: int,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_filtered_markets(markets_dir)
    if df.empty:
        raise SystemExit(f"No finalized markets matched filter under {markets_dir}")

    train, eval_pool = split_train_eval(df)
    eval_index = stratified_sample(eval_pool, per_bucket=per_bucket, seed=seed)

    paths = {
        "markets_filtered": out_dir / "markets_filtered.parquet",
        "train_pool": out_dir / "train_pool.parquet",
        "eval_index": out_dir / "eval_index.parquet",
    }
    df.to_parquet(paths["markets_filtered"], index=False)
    train.to_parquet(paths["train_pool"], index=False)
    eval_index.to_parquet(paths["eval_index"], index=False)

    print(
        f"build_eval_index: {len(df):,} filtered, "
        f"{len(train):,} train pool, {len(eval_index):,} eval rows "
        f"across {eval_index['bucket'].nunique()} buckets"
    )
    return paths


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[2]
    main(
        markets_dir=Path(os.environ.get("EVAL_MARKETS_DIR", here / "data" / "kalshi" / "markets")),
        out_dir=Path(os.environ.get("EVAL_OUT_DIR", here / "eval" / "output")),
        per_bucket=int(os.environ.get("EVAL_PER_BUCKET", "40")),
        seed=int(os.environ.get("EVAL_SEED", "7")),
    )
