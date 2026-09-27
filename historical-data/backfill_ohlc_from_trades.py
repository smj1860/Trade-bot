#!/usr/bin/env python3
"""
Backfills OHLC candles for one symbol by pulling Kraken's raw trade
history and aggregating it — see ohlc_from_trades.py's module docstring
for why this exists (Kraken's OHLC REST endpoint only keeps ~1 month of
history; the Trades endpoint has no such limit, and Kraken's own docs
point at building OHLC from it for exactly this reason).

This writes to the same ohlc_candles table backfill_ohlc.py and
import_csv.py use (same upsert-by-(exchange, symbol, interval, ts) key),
so it's safe to run alongside either of those — re-running over an
already-covered range just updates those rows in place.

Expect this to take a while for high-volume pairs (BTC-USD, ETH-USD),
since the Trades endpoint returns at most 1,000 trades per page — a
low-volume altcoin backfills much faster. This is a deliberate one-time
(or occasional) deep backfill, like import_csv.py, not something to run
on a schedule — run it per-symbol, and expect majors to take hours.

Usage:
    export SUPABASE_DB_URL=postgresql://...
    python3 backfill_ohlc_from_trades.py --symbol BTC-USD --interval 60 --since-days 180

    # A quick timing test on one symbol before committing to the rest:
    python3 backfill_ohlc_from_trades.py --symbol DOGE-USD --interval 60 --since-days 7
"""

from __future__ import annotations

import argparse
import sys
import time

import db
import kraken_client
from ohlc_from_trades import aggregate_trades_to_candles
from symbols import load_symbols

EXCHANGE = "kraken"
FLUSH_EVERY = 500  # candles buffered before an upsert round-trip
PROGRESS_EVERY = 100_000  # trades processed between progress lines


def _counted_trades(trade_stream, symbol: str, started: float):
    """Wraps a trade stream to print periodic progress — this backfill can
    run for hours on a high-volume pair, so a caller watching stderr needs
    some signal it's still moving, not just silence until it finishes."""
    count = 0
    for t in trade_stream:
        count += 1
        if count % PROGRESS_EVERY == 0:
            elapsed_min = (time.time() - started) / 60
            print(f"[{symbol}] ...{count} trades processed ({elapsed_min:.1f}min elapsed)", file=sys.stderr)
        yield t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True, help="Normalized symbol (e.g. BTC-USD).")
    parser.add_argument("--interval", type=int, default=60, help="Candle resolution in minutes to build (default 60).")
    parser.add_argument("--since-days", type=float, default=180.0, help="How far back to pull trade history from now (default 180 = ~6 months).")
    parser.add_argument("--rate-limit-sleep", type=float, default=1.0, help="Seconds to sleep between paginated Trades calls (default 1.0, well under Kraken's public rate limit).")
    args = parser.parse_args()

    all_symbols = load_symbols()
    matches = [s for s in all_symbols if s.symbol == args.symbol]
    if not matches:
        parser.error(f"unknown or disabled symbol: {args.symbol}")
    spec = matches[0]

    now = time.time()
    since_ns = int((now - args.since_days * 86400) * 1_000_000_000)
    print(
        f"[{spec.symbol}] backfilling OHLC (interval={args.interval}min) from raw trades, "
        f"last {args.since_days:.0f} days — this can take a while for high-volume pairs.",
        file=sys.stderr,
    )

    conn = db.connect()
    started = time.time()
    total_candles = 0
    try:
        trade_stream = kraken_client.fetch_trades_window(
            spec.rest_native_symbol, since_ns, now, rate_limit_sleep=args.rate_limit_sleep
        )
        progress_stream = _counted_trades(trade_stream, spec.symbol, started)

        batch: list = []
        earliest = latest = None
        for candle in aggregate_trades_to_candles(progress_stream, args.interval):
            batch.append(candle)
            earliest = candle.ts_unix if earliest is None else min(earliest, candle.ts_unix)
            latest = candle.ts_unix if latest is None else max(latest, candle.ts_unix)
            if len(batch) >= FLUSH_EVERY:
                total_candles += db.upsert_candles(conn, EXCHANGE, spec.symbol, args.interval, batch)
                batch.clear()
        total_candles += db.upsert_candles(conn, EXCHANGE, spec.symbol, args.interval, batch)
        if earliest is not None:
            db.update_backfill_state(conn, EXCHANGE, spec.symbol, args.interval, earliest, latest)
    finally:
        conn.close()

    elapsed_min = (time.time() - started) / 60
    print(f"[{spec.symbol}] done: {total_candles} candles upserted in {elapsed_min:.1f} minutes.", file=sys.stderr)


if __name__ == "__main__":
    main()
