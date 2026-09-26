"""
Postgres connection for the historical data pipeline (Supabase project
Rootstock-vercel, see docs/historical-data-pipeline.md). Connection string
comes ONLY from the SUPABASE_DB_URL environment variable — never a config
file — same pattern as KRAKEN_API_KEY/KRAKEN_API_SECRET elsewhere in this
project, so real credentials are never committed.
"""

from __future__ import annotations

import os

import psycopg2
import psycopg2.extras


class MissingCredentials(RuntimeError):
    pass


def connect():
    dsn = os.environ.get("SUPABASE_DB_URL")
    if not dsn:
        raise MissingCredentials(
            "SUPABASE_DB_URL is not set. Get the connection string from the Supabase "
            "dashboard for the Rootstock-vercel project (Project Settings -> Database "
            "-> Connection string) and export it — never put it in a config file."
        )
    return psycopg2.connect(dsn)


def upsert_candles(conn, exchange: str, symbol: str, interval_minutes: int, candles) -> int:
    """Idempotent insert: re-running a backfill over an already-stored range is a no-op."""
    if not candles:
        return 0
    rows = [
        (
            exchange,
            symbol,
            interval_minutes,
            c.ts_unix,
            c.open,
            c.high,
            c.low,
            c.close,
            c.vwap,
            c.volume,
            c.trade_count,
        )
        for c in candles
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            insert into ohlc_candles
                (exchange, symbol, interval_minutes, ts, open, high, low, close, vwap, volume, trade_count)
            values %s
            on conflict (exchange, symbol, interval_minutes, ts) do update set
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                vwap = excluded.vwap,
                volume = excluded.volume,
                trade_count = excluded.trade_count
            """,
            rows,
            template="(%s, %s, %s, to_timestamp(%s), %s, %s, %s, %s, %s, %s, %s)",
        )
    conn.commit()
    return len(rows)


def upsert_trades(conn, exchange: str, symbol: str, trades) -> int:
    if not trades:
        return 0
    rows = [
        (exchange, symbol, t.trade_id, t.ts_unix_ns, t.price, t.volume, t.side, t.order_type)
        for t in trades
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            insert into trades (exchange, symbol, trade_id, ts, price, volume, side, order_type)
            values %s
            on conflict (exchange, symbol, trade_id) do nothing
            """,
            rows,
            template="(%s, %s, %s, to_timestamp(%s / 1000000000.0), %s, %s, %s, %s)",
        )
    conn.commit()
    return len(rows)


def update_backfill_state(conn, exchange: str, symbol: str, interval_minutes: int, earliest_unix: int, latest_unix: int):
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into ohlc_backfill_state (exchange, symbol, interval_minutes, earliest_ts, latest_ts, last_backfill_at)
            values (%s, %s, %s, to_timestamp(%s), to_timestamp(%s), now())
            on conflict (exchange, symbol, interval_minutes) do update set
                earliest_ts = least(ohlc_backfill_state.earliest_ts, excluded.earliest_ts),
                latest_ts = greatest(ohlc_backfill_state.latest_ts, excluded.latest_ts),
                last_backfill_at = now()
            """,
            (exchange, symbol, interval_minutes, earliest_unix, latest_unix),
        )
    conn.commit()


def get_backfill_state(conn, exchange: str, symbol: str, interval_minutes: int):
    with conn.cursor() as cur:
        cur.execute(
            """
            select extract(epoch from earliest_ts)::bigint, extract(epoch from latest_ts)::bigint
            from ohlc_backfill_state
            where exchange = %s and symbol = %s and interval_minutes = %s
            """,
            (exchange, symbol, interval_minutes),
        )
        row = cur.fetchone()
    return row  # (earliest_unix, latest_unix) or None
