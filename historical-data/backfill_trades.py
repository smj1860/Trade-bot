#!/usr/bin/env python3
"""
Backfills raw trades from Kraken's public REST API into Supabase for one
symbol over a bounded lookback window. Deliberately NOT "backfill full
history for all 13 symbols" by default — see the sizing note in
docs/historical-data-pipeline.md. Run this per-symbol, deliberately, with
a window you've chosen.

Usage:
    export SUPABASE_DB_URL=postgresql://...
    python3 backfill_trades.py --symbol BTC-USD --lookback-hours 24
"""

from __future__ import annotations

import argparse
import sys
import time

import db
import kraken_client
from symbols import load_symbols

EXCHANGE = "kraken"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True, help="Normalized symbol (e.g. BTC-USD).")
    parser.add_argument("--lookback-hours", type=float, default=24.0, help="How far back to pull trades from now.")
    args = parser.parse_args()

    all_symbols = load_symbols()
    matches = [s for s in all_symbols if s.symbol == args.symbol]
    if not matches:
        parser.error(f"unknown or disabled symbol: {args.symbol}")
    spec = matches[0]

    now = time.time()
    since_ns = int((now - args.lookback_hours * 3600) * 1_000_000_000)
    print(f"[{spec.symbol}] backfilling trades for the last {args.lookback_hours}h...", file=sys.stderr)

    conn = db.connect()
    total = 0
    try:
        batch = []
        for trade in kraken_client.fetch_trades_window(spec.rest_native_symbol, since_ns, now):
            batch.append(trade)
            if len(batch) >= 500:
                total += db.upsert_trades(conn, EXCHANGE, spec.symbol, batch)
                batch.clear()
        total += db.upsert_trades(conn, EXCHANGE, spec.symbol, batch)
    finally:
        conn.close()

    print(f"[{spec.symbol}] upserted {total} trades", file=sys.stderr)


if __name__ == "__main__":
    main()
