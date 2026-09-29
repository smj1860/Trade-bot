#!/usr/bin/env python3
"""
Derives coarser-resolution OHLCV candles (120/240/360/720/1440-minute —
2hr/4hr/6hr/12hr/daily) from the 60-minute candles already backfilled by
backfill_ohlc_from_trades.py, instead of re-pulling raw trade history
separately per timeframe.

Why resample instead of re-backfilling from trades at each interval:
Stephen's stated intent (2026-09-28) is to trade live primarily on the
1hr/2hr/4hr/6hr/12hr/daily candles, so the model-training and live
strategy layers both need data at all of these resolutions eventually.
Every higher timeframe here is an exact multiple of 60 minutes, so it can
be built by grouping already-stored hourly candles into bigger buckets —
correct OHLCV math, no approximation — for free, rather than paying the
same hours-per-symbol Kraken Trades-endpoint cost
backfill_ohlc_from_trades.py incurs, once per target interval, on top of
the 60-minute backfill that's already running. 1hr candles themselves are
the --interval 60 rows already in ohlc_candles; this script only produces
the coarser ones.

This writes to the same ohlc_candles table (same upsert-by-(exchange,
symbol, interval, ts) key) as every other backfill script here, so
scripts/train_model.py can use any of these interval_minutes values with
no changes on its side — it already just queries ohlc_candles filtered by
whatever --interval it's given.

Candle aggregation over a bucket of N source candles:
  open   = first source candle's open
  high   = max of source highs
  low    = min of source lows
  close  = last source candle's close
  volume = sum of source volumes
  trade_count = sum of source trade_counts
  vwap   = volume-weighted average of source vwaps (falls back to close
           if the bucket's total volume is zero, same convention
           ohlc_from_trades.py's to_candle() uses)

A target bucket built from fewer than the full N source candles (i.e. a
gap in the underlying hourly data, or a still-forming final bucket at the
edge of what's backfilled so far) is still written — same "corrects
itself on the next re-run" upsert behavior every backfill script here
already relies on rather than trying to detect and skip partial buckets.

Usage:
    export SUPABASE_DB_URL=postgresql://...

    # Resample one symbol to every standard higher timeframe:
    python3 resample_ohlc.py --symbol BTC-USD

    # Resample every configured symbol, only to daily:
    python3 resample_ohlc.py --symbol all --target-intervals 1440

    # Custom target list (must each be a whole multiple of --source-interval):
    python3 resample_ohlc.py --symbol ETH-USD --target-intervals 120,720
"""

from __future__ import annotations

import argparse
import sys

import db
from kraken_client import Candle
from ohlc_from_trades import bucket_start
from symbols import load_symbols

EXCHANGE = "kraken"
DEFAULT_TARGET_INTERVALS = [120, 240, 360, 720, 1440]  # 2hr, 4hr, 6hr, 12hr, daily


def resample_candles(
    candles: list[Candle], source_interval_minutes: int, target_interval_minutes: int
) -> list[Candle]:
    """Pure aggregation logic, kept separate from the DB/CLI glue below so
    it can be unit-tested without a real Postgres connection. `candles`
    must be time-ordered ascending (same convention as every other
    candle/trade stream in this pipeline)."""
    if target_interval_minutes <= source_interval_minutes:
        raise ValueError(
            f"target interval ({target_interval_minutes}min) must be greater than "
            f"the source interval ({source_interval_minutes}min)"
        )
    if target_interval_minutes % source_interval_minutes != 0:
        raise ValueError(
            f"target interval ({target_interval_minutes}min) must be a whole multiple "
            f"of the source interval ({source_interval_minutes}min)"
        )

    buckets: dict[int, list[Candle]] = {}
    for c in candles:
        start = bucket_start(c.ts_unix, target_interval_minutes)
        buckets.setdefault(start, []).append(c)

    resampled = []
    for start in sorted(buckets):
        group = buckets[start]
        total_volume = sum(g.volume for g in group)
        pv_sum = sum(g.vwap * g.volume for g in group)
        vwap = (pv_sum / total_volume) if total_volume else group[-1].close
        resampled.append(
            Candle(
                ts_unix=start,
                open=group[0].open,
                high=max(g.high for g in group),
                low=min(g.low for g in group),
                close=group[-1].close,
                vwap=vwap,
                volume=total_volume,
                trade_count=sum(g.trade_count for g in group),
            )
        )
    return resampled


def fetch_candles(conn, exchange: str, symbol: str, interval_minutes: int) -> list[Candle]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select extract(epoch from ts)::bigint, open, high, low, close, vwap, volume, trade_count
            from ohlc_candles
            where exchange = %s and symbol = %s and interval_minutes = %s
            order by ts asc
            """,
            (exchange, symbol, interval_minutes),
        )
        rows = cur.fetchall()
    return [
        Candle(
            ts_unix=r[0],
            open=r[1],
            high=r[2],
            low=r[3],
            close=r[4],
            vwap=r[5],
            volume=r[6],
            trade_count=r[7],
        )
        for r in rows
    ]


def resample_symbol(conn, symbol: str, source_interval: int, target_intervals: list[int]) -> None:
    source_candles = fetch_candles(conn, EXCHANGE, symbol, source_interval)
    if not source_candles:
        print(f"[{symbol}] no candles at interval={source_interval}min — skipping.", file=sys.stderr)
        return

    for target in target_intervals:
        resampled = resample_candles(source_candles, source_interval, target)
        n = db.upsert_candles(conn, EXCHANGE, symbol, target, resampled)
        print(
            f"[{symbol}] interval={source_interval}min ({len(source_candles)} candles) "
            f"-> interval={target}min: {n} candles upserted",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True, help='Normalized symbol (e.g. "BTC-USD"), or "all".')
    parser.add_argument("--source-interval", type=int, default=60, help="Source candle resolution in minutes (default 60).")
    parser.add_argument(
        "--target-intervals",
        default=",".join(str(i) for i in DEFAULT_TARGET_INTERVALS),
        help="Comma-separated target resolutions in minutes (default: 120,240,360,720,1440 — 2/4/6/12hr and daily).",
    )
    args = parser.parse_args()

    targets = [int(x) for x in args.target_intervals.split(",") if x.strip()]

    conn = db.connect()
    try:
        if args.symbol == "all":
            symbols = [s.symbol for s in load_symbols()]
        else:
            symbols = [args.symbol]
        for symbol in symbols:
            resample_symbol(conn, symbol, args.source_interval, targets)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
