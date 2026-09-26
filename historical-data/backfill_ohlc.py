#!/usr/bin/env python3
"""
Backfills OHLC candles from Kraken's public REST API into Supabase for
the configured symbol universe. See docs/historical-data-pipeline.md for
the retention caveat: 60 (hourly) and 1440 (daily) are the intervals worth
backfilling deeply in one run; finer resolutions only cover what Kraken
currently exposes (recent days, not years) and accumulate real depth by
re-running this script over time (e.g. on a schedule).

Usage:
    export SUPABASE_DB_URL=postgresql://...
    python3 backfill_ohlc.py                       # all enabled symbols, intervals 60 and 1440
    python3 backfill_ohlc.py --symbol BTC-USD       # one symbol only
    python3 backfill_ohlc.py --interval 1440        # one interval only
    python3 backfill_ohlc.py --symbol BTC-USD --interval 1 --since 0   # explicit full-depth attempt
"""

from __future__ import annotations

import argparse
import sys
import time

import db
import kraken_client
from symbols import load_symbols

DEFAULT_INTERVALS = (60, 1440)
EXCHANGE = "kraken"


def backfill_one(conn, symbol: str, rest_native_symbol: str, interval: int, since: int | None) -> None:
    print(f"[{symbol}] interval={interval}min backfilling from since={since}...", file=sys.stderr)
    total = 0
    earliest = None
    latest = None
    batch: list = []

    def flush():
        nonlocal total
        if batch:
            total += db.upsert_candles(conn, EXCHANGE, symbol, interval, batch)
            batch.clear()

    for candle in kraken_client.fetch_ohlc_all(rest_native_symbol, interval, since):
        batch.append(candle)
        earliest = candle.ts_unix if earliest is None else min(earliest, candle.ts_unix)
        latest = candle.ts_unix if latest is None else max(latest, candle.ts_unix)
        if len(batch) >= 500:
            flush()
    flush()

    if earliest is not None and latest is not None:
        db.update_backfill_state(conn, EXCHANGE, symbol, interval, earliest, latest)

    print(f"[{symbol}] interval={interval}min: upserted {total} candles", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", help="Normalized symbol (e.g. BTC-USD). Default: all enabled symbols.")
    parser.add_argument(
        "--interval",
        type=int,
        choices=kraken_client.VALID_INTERVALS,
        help="Kraken OHLC interval in minutes. Default: backfill both 60 and 1440.",
    )
    parser.add_argument(
        "--since",
        type=int,
        default=0,
        help="Unix timestamp to backfill from (default: 0, i.e. as far back as Kraken has for this interval).",
    )
    args = parser.parse_args()

    all_symbols = load_symbols()
    if args.symbol:
        symbols = [s for s in all_symbols if s.symbol == args.symbol]
        if not symbols:
            parser.error(f"unknown or disabled symbol: {args.symbol}")
    else:
        symbols = all_symbols

    intervals = [args.interval] if args.interval else list(DEFAULT_INTERVALS)

    conn = db.connect()
    try:
        for s in symbols:
            for interval in intervals:
                backfill_one(conn, s.symbol, s.rest_native_symbol, interval, args.since)
                time.sleep(1)  # be polite to Kraken's public rate limit across symbols too
    finally:
        conn.close()


if __name__ == "__main__":
    main()
