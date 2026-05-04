# Eval Pipeline

Builds a curated, stratified eval set from Kalshi `markets/` parquet, computes
several baselines (HOLD, BUY YES, BUY NO, price-as-prob, EV-greedy) and writes
a Markdown report to `eval/output/REPORT.md`.

## Layout

```
eval/
  src/
    build_eval_index.py    # filter + stratify -> eval_index.parquet
    calibration.py         # price -> empirical win rate (training half)
    baselines.py           # baseline policies and PnL math
    run_baselines.py       # run baselines on eval_index, write metrics
    score_report.py        # render REPORT.md from baseline metrics
    pipeline.py            # orchestrator (runs the four steps in order)
    build_time_eval_index.py
                             # ticker-level first-half trade-context tasks
    run_time_model.py        # OpenRouter runner for time-based eval
    time_eval_scoring.py     # P(YES) + trade scoring
  output/                  # generated artifacts (CSV / parquet / json / md)
```

## Run it

From the project root:

```bash
uv run python -m eval.src.pipeline
```

Run a model through OpenRouter:

```bash
PYTHONPATH="$PWD" python -m eval.src.run_model --model z-ai/glm-5.1 --num-examples 5 --max-tokens 2048
```

Estimate tokens and expected OpenRouter cost before making API calls:

```bash
PYTHONPATH="$PWD" python -m eval.src.run_model --model z-ai/glm-5.1 --num-examples 5 --max-tokens 2048 --estimate-only
```

Abort before running if the estimated maximum cost is too high:

```bash
PYTHONPATH="$PWD" python -m eval.src.run_model --model z-ai/glm-5.1 --num-examples 5 --max-tokens 2048 --max-estimated-cost 0.05
```

## Time-Based Eval

The time-based eval compares LLM forecasting accuracy against the market price
baseline, using `trades/*.parquet` for pre-freeze price history and
`markets/*.parquet` for metadata plus the hidden resolved outcome.

Each eval item is a `(market, freeze_time)` row:

```python
eval_item = {
    "market_id": "...",
    "question_text": "...",
    "resolution_criteria": "...",
    "t0": "...",                       # freeze time
    "price_history": "trades <= t0",    # what the model sees
    "market_price_at_t0": 0.63,         # baseline forecast
    "resolution_date": "...",
    "resolution": 0 or 1,
    "horizon_days": 12.3,
    "category": "sports/econ/...",
    "liquidity_tier": "low/mid/high",
}
```

Freeze-time sampling:

- Keep markets with lifetime at least 7 days.
- Sample `t0` at `10%`, `25%`, `50%`, `75%`, and `90%` of market lifetime.
- Skip freeze times within 24 hours of resolution.
- Bucket items into `short`, `mid`, and `long` by `horizon_days` terciles.
- Stratify sampling by horizon bucket and category.
- Use a recent pre-freeze VWAP as the market-price baseline and keep only
  non-trivial freeze points with baseline `P(YES)` in `[0.15, 0.85]`.
- Include every pre-freeze trade row by default; set
  `TIME_EVAL_RECENT_TRADES` to a positive number to cap the table.

Build the time eval index:

```bash
uv run python -m eval.src.build_time_eval_index
```

Run a model on balanced horizon/category time-context tasks:

```bash
PYTHONPATH="$PWD" python -m eval.src.run_time_model --model z-ai/glm-5.1 --num-examples 9 --max-tokens 256
```

Estimate OpenRouter cost without model calls:

```bash
PYTHONPATH="$PWD" python -m eval.src.run_time_model --model z-ai/glm-5.1 --num-examples 9 --max-tokens 256 --estimate-only
```

Each prompt asks only for a probability forecast:

```text
P(YES)=0.63
```

The evaluator derives any simulated trade deterministically from the forecast's
edge versus the market price baseline.

Override defaults via env vars:

- `EVAL_MARKETS_DIR` (default: `data/kalshi/markets`)
- `EVAL_OUT_DIR`     (default: `eval/output`)
- `EVAL_PER_BUCKET`  (default: `40`) — markets per price bucket
- `EVAL_SEED`        (default: `7`)
- `TIME_EVAL_RECENT_TRADES` (default: `0`) — include all pre-freeze trades; positive values cap the table
- `TIME_EVAL_BASELINE_HOURS` (default: `24`) — recent VWAP window for the market-price baseline

## Outputs

- `eval/output/markets_filtered.parquet`
- `eval/output/eval_index.parquet`
- `eval/output/calibration_curve.csv`
- `eval/output/baseline_results.parquet`
- `eval/output/baseline_summary.csv`
- `eval/output/token_estimate__<provider>__<model>.json`
- `eval/output/model_summary.json` (includes estimated and actual token/cost accounting after model runs)
- `eval/output/REPORT.md`

Time eval outputs:

- `eval/output/time_eval/time_eval_index.parquet`
- `eval/output/time_eval/time_token_estimate__<provider>__<model>.json`
- `eval/output/time_eval/time_model_results__<provider>__<model>.jsonl`
- `eval/output/time_eval/time_model_scored__<provider>__<model>.csv`
- `eval/output/time_eval/time_model_summary__<provider>__<model>.csv`
- `eval/output/time_eval/time_model_metadata__<provider>__<model>.json`
