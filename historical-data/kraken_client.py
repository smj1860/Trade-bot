"""
Minimal Kraken public REST client for historical data: OHLC candles and raw
trades. Deliberately separate from rust-core/src/kraken_rest.rs — that
client signs authenticated requests for order execution; this one only
ever calls Kraken's public, unauthenticated endpoints, and lives in Python
alongside the rest of this offline batch pipeline (see
docs/historical-data-pipeline.md for why this isn't in Rust).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal

import requests

KRAKEN_REST_URL = "https://api.kraken.com"

# Kraken's supported OHLC candle resolutions, in minutes.
VALID_INTERVALS = (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)


class KrakenApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class Candle:
    ts_unix: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    vwap: Decimal
    volume: Decimal
    trade_count: int


@dataclass(frozen=True)
class Trade:
    trade_id: int
    ts_unix_ns: int
    price: Decimal
    volume: Decimal
    side: str  # "buy" | "sell"
    order_type: str  # "market" | "limit"


def _get(path: str, params: dict, *, timeout: float = 15.0) -> dict:
    resp = requests.get(f"{KRAKEN_REST_URL}{path}", params=params, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    if body.get("error"):
        raise KrakenApiError(f"{path} {params}: {body['error']}")
    return body["result"]


def fetch_ohlc(pair: str, interval: int, since: int | None = None) -> tuple[list[Candle], int]:
    """
    One page of OHLC candles for `pair` at `interval` minutes, starting
    just after `since` (a unix timestamp) if given. Kraken returns up to
    720 candles per call and a `last` cursor for the next page — see the
    retention caveat in docs/historical-data-pipeline.md: `since` only
    reaches back as far as Kraken still has data for that interval, which
    for fine resolutions (1/5/15 min) is nowhere near "years".

    Returns (candles, next_since) — call again with next_since to page
    forward; stop once next_since stops advancing (caught up to now) or a
    call returns zero new candles.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval must be one of {VALID_INTERVALS}, got {interval}")
    params: dict = {"pair": pair, "interval": interval}
    if since is not None:
        params["since"] = since
    result = _get("/0/public/OHLC", params)

    # The pair key in the response isn't always exactly what was requested
    # (Kraken sometimes echoes its own internal/legacy key) — "last" is the
    # only other top-level key, so whichever key isn't "last" is the data.
    data_key = next(k for k in result if k != "last")
    candles = [
        Candle(
            ts_unix=int(row[0]),
            open=Decimal(row[1]),
            high=Decimal(row[2]),
            low=Decimal(row[3]),
            close=Decimal(row[4]),
            vwap=Decimal(row[5]),
            volume=Decimal(row[6]),
            trade_count=int(row[7]),
        )
        for row in result[data_key]
    ]
    return candles, int(result["last"])


def fetch_ohlc_all(pair: str, interval: int, since: int | None = None, *, rate_limit_sleep: float = 1.0):
    """
    Pages through fetch_ohlc until caught up to now, yielding candles as
    they arrive. `rate_limit_sleep` between calls keeps this well under
    Kraken's public-endpoint rate limit.
    """
    cursor = since
    while True:
        candles, next_cursor = fetch_ohlc(pair, interval, cursor)
        if not candles:
            return
        yield from candles
        if next_cursor == cursor:
            return
        cursor = next_cursor
        time.sleep(rate_limit_sleep)


def fetch_trades(pair: str, since: str | int | None = None) -> tuple[list[Trade], str]:
    """
    One page of raw trades (up to 1000) for `pair`, starting just after
    `since` (Kraken's own opaque cursor — an integer nanosecond timestamp
    as a string; pass the `next_since` from a previous call, or omit for
    the most recent trades). Returns (trades, next_since).
    """
    params: dict = {"pair": pair}
    if since is not None:
        params["since"] = str(since)
    result = _get("/0/public/Trades", params)

    data_key = next(k for k in result if k != "last")
    trades = [
        Trade(
            trade_id=int(row[6]),
            ts_unix_ns=int(Decimal(row[2]) * Decimal(1_000_000_000)),
            price=Decimal(row[0]),
            volume=Decimal(row[1]),
            side="buy" if row[3] == "b" else "sell",
            order_type="market" if row[4] == "m" else "limit",
        )
        for row in result[data_key]
    ]
    return trades, result["last"]


def fetch_trades_window(
    pair: str,
    since: str | int,
    until_unix: float,
    *,
    rate_limit_sleep: float = 1.0,
):
    """
    Pages through fetch_trades starting at `since`, yielding trades until
    a page's trades pass `until_unix` (a unix timestamp — typically
    time.time() at call time) or a page comes back empty. Bounded on
    purpose: see the "not an unbounded pull" note in
    docs/historical-data-pipeline.md.
    """
    cursor = since
    while True:
        trades, next_cursor = fetch_trades(pair, cursor)
        if not trades:
            return
        for t in trades:
            if t.ts_unix_ns / 1_000_000_000 > until_unix:
                return
            yield t
        if next_cursor == cursor:
            return
        cursor = next_cursor
        time.sleep(rate_limit_sleep)
