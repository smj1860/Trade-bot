"""Quick DB depth check for the historical OHLC backfill.

For every configured symbol, compares what ``ohlc_backfill_state`` claims with
what is actually in ``ohlc_candles`` and reports depth in days against a target.

Usage (needs SUPABASE_DB_URL):
    python check_backfill_depth.py [--interval 60] [--target-days 1080]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Optional

EXCHANGE = "kraken"
DAY = 86400


@dataclass
class DepthRow:
    symbol: str
    state_earliest: Optional[int]
    state_latest: Optional[int]
    candle_min: Optional[int]
    candle_max: Optional[int]
    candle_count: int


def depth_days(earliest_unix: Optional[int], now: int) -> Optional[float]:
    if earliest_unix is None:
        return None
    return (now - earliest_unix) / DAY


def expected_candles(min_ts: Optional[int], max_ts: Optional[int], interval_minutes: int) -> int:
    if min_ts is None or max_ts is None:
        return 0
    return int((max_ts - min_ts) // (interval_minutes * 60)) + 1


def classify(row: DepthRow, now: int, target_days: int, interval_minutes: int) -> str:
    d = depth_days(row.candle_min, now)
    if d is None:
        return "EMPTY"
    if d >= target_days - 1:
        return "OK"
    return "SHORT"


def format_row(row: DepthRow, now: int, target_days: int, interval_minutes: int) -> str:
    d = depth_days(row.candle_min, now)
    sd = depth_days(row.state_earliest, now)
    exp = expected_candles(row.candle_min, row.candle_max, interval_minutes)
    cov = (row.candle_count / exp * 100) if exp else 0.0
    return (
        f"{row.symbol:<10} {classify(row, now, target_days, interval_minutes):<6} "
        f"db_depth={'-' if d is None else f'{d:7.1f}'}d "
        f"state_depth={'-' if sd is None else f'{sd:7.1f}'}d "
        f"candles={row.candle_count:>8} expected={exp:>8} coverage={cov:5.1f}%"
    )


def fetch_row(conn, symbol: str, interval_minutes: int) -> DepthRow:
    import db

    state = db.get_backfill_state(conn, EXCHANGE, symbol, interval_minutes)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MIN(ts), MAX(ts), COUNT(*) FROM ohlc_candles "
            "WHERE exchange = %s AND symbol = %s AND interval_minutes = %s",
            (EXCHANGE, symbol, interval_minutes),
        )
        mn, mx, cnt = cur.fetchone()
    return DepthRow(
        symbol=symbol,
        state_earliest=state[0] if state else None,
        state_latest=state[1] if state else None,
        candle_min=_to_unix(mn),
        candle_max=_to_unix(mx),
        candle_count=int(cnt or 0),
    )


def _to_unix(v) -> Optional[int]:
    if v is None:
        return None
    if hasattr(v, "timestamp"):
        return int(v.timestamp())
    return int(v)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--target-days", type=int, default=1080)
    args = p.parse_args(argv)

    import db
    from symbols import load_symbols

    now = int(time.time())
    conn = db.connect()
    try:
        rows = [fetch_row(conn, s.symbol, args.interval) for s in load_symbols()]
    finally:
        conn.close()

    for r in rows:
        print(format_row(r, now, args.target_days, args.interval))
    short = [r.symbol for r in rows if classify(r, now, args.target_days, args.interval) != "OK"]
    print(f"\n{len(rows) - len(short)}/{len(rows)} at >= {args.target_days}d; short: {', '.join(short) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
