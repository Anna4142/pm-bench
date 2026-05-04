"""Build freeze-time prediction-market eval tasks.

Each row is one ``(market, freeze_time)`` item. The model sees market metadata
and every trade before the freeze time; the final outcome and future trades stay
hidden for scoring. The market price at freeze time is stored as the baseline
forecast.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import pandas as pd

SYSTEM_PROMPT = """You are evaluating forecasting accuracy for a historical prediction market at a freeze time.

You are shown only market information and trades that happened before the freeze time.
The final outcome and all future trades are hidden. Estimate P(YES).

Your entire response must be exactly one line and the first token must be P(YES)=.
Do not explain, justify, or include any other text.

Required format:
P(YES)=<number between 0 and 1>

Do not output a trade. The evaluator will deterministically derive any simulated trade from your forecast and the market price baseline."""

FREEZE_FRACTIONS: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
MIN_LIFETIME_DAYS = 7.0
MIN_HORIZON_HOURS = 24.0
MARKET_PRICE_MIN = 0.15
MARKET_PRICE_MAX = 0.85


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_default(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _bucket_by_quantiles(
    df: pd.DataFrame,
    column: str,
    output_column: str,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    labels = labels or ["short", "mid", "long"]
    df = df.copy()
    if len(df) < 3:
        df[output_column] = labels[: len(df)]
        return df
    ranked = df[column].rank(method="first")
    df[output_column] = pd.qcut(ranked, q=3, labels=labels)
    df[output_column] = df[output_column].astype(str)
    return df


def _sample_by_bucket(df: pd.DataFrame, per_bucket: int, seed: int) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for _, group in df.groupby(["horizon_bucket", "category"], sort=False):
        chunks.append(group.sample(n=min(per_bucket, len(group)), random_state=seed))
    return pd.concat(chunks, ignore_index=True).sort_values(["horizon_bucket", "category", "ticker", "freeze_fraction"])


def _classify_category(title: str, event_ticker: str, market_type: str) -> str:
    text = f"{title} {event_ticker} {market_type}".lower()
    if any(x in text for x in ("pres", "senate", "house", "election", "trump", "biden", "kamala", "mayor")):
        return "politics"
    if any(x in text for x in ("fed", "cpi", "gdp", "unemployment", "inflation", "trade deficit", "interest rate")):
        return "econ"
    if any(x in text for x in ("nba", "nfl", "nhl", "mlb", "game", "series winner", "championship", "win the")):
        return "sports"
    if any(x in text for x in ("temp", "temperature", "weather", "rain", "snow", "hurricane")):
        return "weather"
    if any(x in text for x in ("bitcoin", "ethereum", "crypto", "stock", "s&p", "nasdaq", "price")):
        return "finance"
    if any(x in text for x in ("movie", "album", "rotten tomatoes", "grammy", "oscar", "box office")):
        return "culture"
    if any(x in text for x in ("covid", "vaxx", "vaccine", "cases")):
        return "health"
    return "other"


def _format_recent_trades(trades: list[dict]) -> str:
    if not trades:
        return "(no pre-freeze trades)"
    lines = ["time | YES price | NO price | contracts | taker side"]
    for trade in trades:
        lines.append(
            f"{trade['created_time']} | {trade['yes_price']}c | {trade['no_price']}c | "
            f"{trade['count']} | {trade['taker_side']}"
        )
    return "\n".join(lines)


def _build_prompt(row: pd.Series) -> str:
    price_history = json.loads(row["price_history_json"])
    return f"""Task: Forecast a prediction market at the freeze time and compare your belief to the market price.

Market:
Market ID: {row['market_id']}
Title: {row['title']}
Resolution criteria: {row['resolution_criteria']}
Ticker: {row['ticker']}
Market created: {row['market_created_time']}
Resolution date: {row['resolution_date']}

Timeline:
First trade: {row['first_trade']}
Freeze time t0: {row['t0']}
Horizon to resolution: {row['horizon_days']:.2f} days

Market price baseline at t0:
YES price: {row['market_price_at_t0']:.2f}
NO price: {1.0 - row['market_price_at_t0']:.2f}
Recent baseline window trades: {int(row['recent_baseline_trades'])}

Pre-freeze price history summary:
Number of pre-freeze trades: {int(row['context_trades'])}
Pre-freeze contracts: {int(row['context_contracts'])}
First YES price: {row['first_yes_price']}c
Last YES price: {row['last_yes_price']}c
YES price change: {row['yes_price_change']}c
VWAP YES price: {row['vwap_yes_price']:.2f}c
Taker YES trades: {int(row['taker_yes_trades'])}
Taker NO trades: {int(row['taker_no_trades'])}

Full pre-freeze price history:
{_format_recent_trades(price_history)}

Return exactly one line now: P(YES)=<number between 0 and 1>"""


def _load_candidate_spans(
    con: duckdb.DuckDBPyConnection,
    markets_dir: Path,
    trades_dir: Path,
    min_total_trades: int,
) -> pd.DataFrame:
    markets_glob = str(markets_dir / "*.parquet")
    trades_glob = str(trades_dir / "*.parquet")
    return con.execute(
        f"""
        WITH resolved_markets AS (
            SELECT
                ticker,
                any_value(event_ticker) AS event_ticker,
                any_value(market_type) AS market_type,
                any_value(title) AS title,
                any_value(yes_sub_title) AS yes_sub_title,
                any_value(no_sub_title) AS no_sub_title,
                any_value(result) AS result,
                min(created_time) AS market_created_time,
                max(close_time) AS resolution_date,
                max(volume) AS volume,
                max(open_interest) AS open_interest
            FROM '{markets_glob}'
            WHERE status = 'finalized'
              AND result IN ('yes', 'no')
              AND title IS NOT NULL
              AND close_time IS NOT NULL
            GROUP BY ticker
        ),
        trade_spans AS (
            SELECT
                ticker,
                min(created_time) AS first_trade,
                max(created_time) AS last_trade,
                count(*) AS total_trades,
                sum(count) AS total_contracts
            FROM '{trades_glob}'
            WHERE created_time IS NOT NULL
              AND yes_price BETWEEN 1 AND 99
              AND no_price BETWEEN 1 AND 99
            GROUP BY ticker
            HAVING count(*) >= {min_total_trades}
        )
        SELECT
            m.ticker,
            m.event_ticker,
            m.market_type,
            m.title,
            m.yes_sub_title,
            m.no_sub_title,
            m.result,
            m.market_created_time,
            m.resolution_date,
            m.volume,
            m.open_interest,
            s.first_trade,
            s.last_trade,
            s.total_trades,
            s.total_contracts,
            epoch(m.resolution_date) - epoch(s.first_trade) AS lifetime_seconds
        FROM trade_spans s
        INNER JOIN resolved_markets m USING (ticker)
        WHERE m.resolution_date > s.first_trade
        """
    ).df()


def _load_context_for_sample(
    con: duckdb.DuckDBPyConnection,
    trades_dir: Path,
    sample: pd.DataFrame,
    min_context_trades: int,
    recent_trades: int,
    baseline_hours: int,
) -> pd.DataFrame:
    trades_glob = str(trades_dir / "*.parquet")
    sample_for_duck = sample[["example_id", "ticker", "t0"]].copy()
    con.register("sample_tickers", sample_for_duck)
    summary = con.execute(
        f"""
        WITH visible AS (
            SELECT
                s.example_id,
                s.t0,
                t.ticker,
                t.created_time,
                t.yes_price,
                t.no_price,
                t.count,
                t.taker_side
            FROM '{trades_glob}' t
            INNER JOIN sample_tickers s
                ON t.ticker = s.ticker
               AND t.created_time <= s.t0
            WHERE t.yes_price BETWEEN 1 AND 99
              AND t.no_price BETWEEN 1 AND 99
        )
        SELECT
            example_id,
            any_value(ticker) AS ticker,
            count(*) AS context_trades,
            sum(count) AS context_contracts,
            arg_min(yes_price, created_time) AS first_yes_price,
            arg_max(yes_price, created_time) AS last_yes_price,
            arg_max(no_price, created_time) AS last_no_price,
            sum(yes_price * count) / nullif(sum(count), 0) AS vwap_yes_price,
            sum(
                CASE
                    WHEN created_time >= t0 - INTERVAL '{baseline_hours} hours'
                    THEN yes_price * count
                    ELSE 0
                END
            ) / nullif(
                sum(
                    CASE
                        WHEN created_time >= t0 - INTERVAL '{baseline_hours} hours'
                        THEN count
                        ELSE 0
                    END
                ),
                0
            ) AS recent_vwap_yes_price,
            sum(
                CASE
                    WHEN created_time >= t0 - INTERVAL '{baseline_hours} hours'
                    THEN 1
                    ELSE 0
                END
            ) AS recent_baseline_trades,
            min(created_time) AS first_visible_trade_time,
            max(created_time) AS last_visible_trade_time,
            sum(CASE WHEN taker_side = 'yes' THEN 1 ELSE 0 END) AS taker_yes_trades,
            sum(CASE WHEN taker_side = 'no' THEN 1 ELSE 0 END) AS taker_no_trades
        FROM visible
        GROUP BY example_id
        HAVING count(*) >= {min_context_trades}
        """
    ).df()

    valid_examples = summary[["example_id"]]
    con.register("valid_examples", valid_examples)
    trade_limit_predicate = "" if recent_trades <= 0 else f"WHERE rn <= {recent_trades}"
    recent = con.execute(
        f"""
        WITH visible AS (
            SELECT
                s.example_id,
                t.ticker,
                t.created_time,
                t.yes_price,
                t.no_price,
                t.count,
                t.taker_side,
                row_number() OVER (PARTITION BY s.example_id ORDER BY t.created_time DESC) AS rn
            FROM '{trades_glob}' t
            INNER JOIN sample_tickers s
                ON t.ticker = s.ticker
               AND t.created_time <= s.t0
            INNER JOIN valid_examples v
                ON s.example_id = v.example_id
            WHERE t.yes_price BETWEEN 1 AND 99
              AND t.no_price BETWEEN 1 AND 99
        )
        SELECT example_id, ticker, created_time, yes_price, no_price, count, taker_side
        FROM visible
        {trade_limit_predicate}
        ORDER BY example_id, created_time
        """
    ).df()

    recent = recent.assign(created_time=recent["created_time"].astype(str))
    recent_json = pd.DataFrame(
        [
            {
                "example_id": example_id,
                "price_history_json": json.dumps(
                    group.drop(columns=["example_id", "ticker"]).to_dict("records"),
                    default=_json_default,
                ),
            }
            for example_id, group in recent.groupby("example_id")
        ]
    )
    return summary.merge(recent_json, on="example_id", how="left")


def build_time_eval_index(
    markets_dir: Path,
    trades_dir: Path,
    out_dir: Path,
    per_bucket: int = 20,
    seed: int = 7,
    min_total_trades: int = 20,
    min_context_trades: int = 10,
    recent_trades: int = 0,
    baseline_hours: int = 24,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        spans = _load_candidate_spans(con, markets_dir, trades_dir, min_total_trades)
        if spans.empty:
            raise SystemExit("No eligible tickers found for time eval.")

        spans["first_trade"] = pd.to_datetime(spans["first_trade"], utc=True)
        spans["last_trade"] = pd.to_datetime(spans["last_trade"], utc=True)
        spans["market_created_time"] = pd.to_datetime(spans["market_created_time"], utc=True, errors="coerce")
        spans["resolution_date"] = pd.to_datetime(spans["resolution_date"], utc=True, errors="coerce")
        spans["lifetime_days"] = spans["lifetime_seconds"] / 86_400.0
        spans = spans[spans["lifetime_days"] >= MIN_LIFETIME_DAYS].copy()
        if spans.empty:
            raise SystemExit(f"No eligible tickers have lifetime >= {MIN_LIFETIME_DAYS} days.")

        spans["category"] = spans.apply(
            lambda row: _classify_category(str(row["title"]), str(row["event_ticker"]), str(row["market_type"])),
            axis=1,
        )
        spans["resolution_criteria"] = (
            "YES: "
            + spans["yes_sub_title"].fillna("wins")
            + " | NO: "
            + spans["no_sub_title"].fillna("does not win")
        )
        spans["liquidity_value"] = spans["total_contracts"].fillna(spans["volume"]).fillna(0)
        spans = _bucket_by_quantiles(spans, "liquidity_value", "liquidity_tier", labels=["low", "mid", "high"])

        freeze_items: list[pd.DataFrame] = []
        for fraction in FREEZE_FRACTIONS:
            item = spans.copy()
            item["freeze_fraction"] = fraction
            item["t0"] = item["first_trade"] + (item["resolution_date"] - item["first_trade"]) * fraction
            item["horizon_days"] = (item["resolution_date"] - item["t0"]).dt.total_seconds() / 86_400.0
            freeze_items.append(item)
        items = pd.concat(freeze_items, ignore_index=True)
        items = items[items["horizon_days"] >= (MIN_HORIZON_HOURS / 24.0)].copy()
        if items.empty:
            raise SystemExit("No freeze-time items remain after horizon filter.")

        items = _bucket_by_quantiles(items, "horizon_days", "horizon_bucket")
        items["span_bucket"] = items["horizon_bucket"]
        items = items.sort_values(["horizon_bucket", "category", "ticker", "freeze_fraction"]).reset_index(drop=True)
        items.insert(0, "example_id", range(len(items)))
        # Oversample before context-dependent filters so uncertainty and
        # min-trade filters do not empty small strata too aggressively.
        sample = _sample_by_bucket(items, per_bucket=per_bucket * 5, seed=seed)

        context = _load_context_for_sample(
            con,
            trades_dir,
            sample,
            min_context_trades=min_context_trades,
            recent_trades=recent_trades,
            baseline_hours=baseline_hours,
        )
    finally:
        con.close()

    index = sample.merge(context, on=["example_id", "ticker"], how="inner")
    index["price_history_json"] = index["price_history_json"].fillna("[]")
    index["market_id"] = index["ticker"]
    index["question_text"] = index["title"]
    index["resolution"] = (index["result"] == "yes").astype(int)
    index["recent_vwap_yes_price"] = index["recent_vwap_yes_price"].fillna(index["vwap_yes_price"])
    index["recent_baseline_trades"] = index["recent_baseline_trades"].fillna(0)
    index["market_price_at_t0"] = index["recent_vwap_yes_price"] / 100.0
    index["baseline_prob_yes"] = index["market_price_at_t0"]
    index["yes_price_change"] = index["last_yes_price"] - index["first_yes_price"]
    index = index[index["market_price_at_t0"].between(MARKET_PRICE_MIN, MARKET_PRICE_MAX)].copy()
    if index.empty:
        raise SystemExit("No sampled tasks remain after market-price uncertainty filter.")
    index = _sample_by_bucket(index, per_bucket=per_bucket, seed=seed)
    index = index.sort_values(["horizon_bucket", "category", "ticker", "freeze_fraction"]).reset_index(drop=True)
    index["prompt"] = index.apply(_build_prompt, axis=1)

    path = out_dir / "time_eval_index.parquet"
    index.to_parquet(path, index=False)
    print(
        f"build_time_eval_index: {len(spans):,} eligible markets, "
        f"{len(items):,} freeze items, {len(index):,} sampled tasks -> {path}"
    )
    print(index.groupby(["horizon_bucket", "category"]).size().to_string())
    return index


def main() -> None:
    root = _default_root()
    build_time_eval_index(
        markets_dir=Path(os.environ.get("EVAL_MARKETS_DIR", root / "data" / "kalshi" / "markets")),
        trades_dir=Path(os.environ.get("EVAL_TRADES_DIR", root / "data" / "kalshi" / "trades")),
        out_dir=Path(os.environ.get("TIME_EVAL_OUT_DIR", root / "eval" / "output" / "time_eval")),
        per_bucket=int(os.environ.get("TIME_EVAL_PER_BUCKET", "20")),
        seed=int(os.environ.get("EVAL_SEED", "7")),
        min_total_trades=int(os.environ.get("TIME_EVAL_MIN_TOTAL_TRADES", "20")),
        min_context_trades=int(os.environ.get("TIME_EVAL_MIN_CONTEXT_TRADES", "10")),
        recent_trades=int(os.environ.get("TIME_EVAL_RECENT_TRADES", "0")),
        baseline_hours=int(os.environ.get("TIME_EVAL_BASELINE_HOURS", "24")),
    )


if __name__ == "__main__":
    main()
