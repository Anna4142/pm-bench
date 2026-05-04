"""End-to-end eval pipeline.

Run via:

    uv run python -m eval.src.pipeline

Steps:
  1. build_eval_index.main(...)
  2. calibration.main(...)
  3. run_baselines.main(...)
  4. score_report.main(...)

All artifacts land in ``$EVAL_OUT_DIR`` (defaults to ``eval/output/``).
"""

from __future__ import annotations

import os
from pathlib import Path

from eval.src import build_eval_index, calibration, run_baselines, score_report


def main() -> None:
    here = Path(__file__).resolve().parents[2]
    markets_dir = Path(os.environ.get("EVAL_MARKETS_DIR", here / "data" / "kalshi" / "markets"))
    out_dir = Path(os.environ.get("EVAL_OUT_DIR", here / "eval" / "output"))
    per_bucket = int(os.environ.get("EVAL_PER_BUCKET", "40"))
    seed = int(os.environ.get("EVAL_SEED", "7"))

    print(f"== eval pipeline ==\n  markets_dir: {markets_dir}\n  out_dir: {out_dir}")

    build_eval_index.main(markets_dir=markets_dir, out_dir=out_dir, per_bucket=per_bucket, seed=seed)
    calibration.main(out_dir=out_dir)
    run_baselines.main(out_dir=out_dir)
    score_report.main(out_dir=out_dir)

    print("\nDone. See:")
    for name in (
        "markets_filtered.parquet",
        "eval_index.parquet",
        "calibration_curve.csv",
        "baseline_results.parquet",
        "baseline_summary.csv",
        "baseline_summary_by_bucket.csv",
        "REPORT.md",
    ):
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
