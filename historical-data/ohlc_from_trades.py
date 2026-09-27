"""
Aggregates Kraken's raw historical trades (from the public Trades REST
endpoint) into OHLCVT candles — the same shape backfill_ohlc.py and
import_csv.py write into the ohlc_candles table.

Why this exists: Kraken's OHLC REST endpoint (used by backfill_ohlc.py)
only ever returns the most recent 720 candles per interval — at hourly
resolution, that's ~1 month — which is the wall every round of
docs/model-training.md has run into. Kraken's own support docs confirm
there's no deeper-history OHLC endpoint, but point at an alternative: "For
applications that require additional OHLC or tick data, it is possible to
retrieve the entire trading history of our markets ... via the REST API
Trades endpoint. The OHLC for any time frame and any interval can then be
created from the historical time and sales data." Passing since=0 (or any
timestamp) to the Trades endpoint pages through the *entire* trade
history for a pair, with no retention limit — kraken_client.py's
fetch_trades()/fetch_trades_window() already do this paging. This module
is the missing piece: turning that raw trade stream into OHLC candles,
so the existing training pipeline (which reads from ohlc_candles, not raw
trades) can use it without any changes on that side.

Tradeoff versus historical-data/import_csv.py's Kraken zip-dump import:
same end result (deep historical OHLC data) with no file download/
extraction to manage, but the Trades endpoint returns at most 1,000
trades per page — for a high-volume pair (BTC-USD), pulling months of
history means many thousands of paginated requests and can take hours;
a low-volume altcoin is fast. See backfill_ohlc_from_trades.py for the
script that drives this against a real symbol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable, Iterator

from kraken_client import Candle, Trade


def bucket_start(ts_unix: float, interval_minutes: int) -> int:
    """The unix-second timestamp of the start of the interval-minute
    bucket containing ts_unix — e.g. bucket_start(3723, 60) == 3600 (the
    top of that hour). Buckets are aligned to wall-clock interval
    boundaries from the unix epoch (00:00:00 UTC), the same convention
    Kraken's own OHLC candles use, not to whenever the first trade in a
    window happens to land."""
    interval_seconds = interval_minutes * 60
    return int(ts_unix // interval_seconds) * interval_seconds


@dataclass
class _OpenBucket:
    ts_unix: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal = field(default_factory=lambda: Decimal(0))
    trade_count: int = 0
    _pv_sum: Decimal = field(default_factory=lambda: Decimal(0))  # price * volume, for a volume-weighted vwap

    def add(self, trade: Trade) -> None:
        self.high = max(self.high, trade.price)
        self.low = min(self.low, trade.price)
        self.close = trade.price
        self.volume += trade.volume
        self.trade_count += 1
        self._pv_sum += trade.price * trade.volume

    def to_candle(self) -> Candle:
        vwap = (self._pv_sum / self.volume) if self.volume else self.close
        return Candle(
            ts_unix=self.ts_unix,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            vwap=vwap,
            volume=self.volume,
            trade_count=self.trade_count,
        )


def aggregate_trades_to_candles(trades: Iterable[Trade], interval_minutes: int) -> Iterator[Candle]:
    """Buckets a time-ordered stream of trades into OHLCVT candles at
    `interval_minutes` resolution, yielding each candle as soon as a
    trade from the *next* bucket is seen — this can run as a streaming
    generator over a paginated trade fetch without ever holding the
    whole history in memory. The final, still-open bucket is also
    yielded once the input is exhausted; callers pulling a window that
    ends "now" should treat that last candle as potentially a partial
    (still-forming) bucket, same as any live exchange's most recent
    candle — re-running the backfill later naturally corrects it via the
    same (exchange, symbol, interval, ts) upsert every other backfill
    script here uses.

    Requires `trades` to be time-ordered (ascending) — exactly what
    kraken_client.fetch_trades()/fetch_trades_window() yield."""
    current: _OpenBucket | None = None
    for trade in trades:
        ts = trade.ts_unix_ns / 1_000_000_000
        start = bucket_start(ts, interval_minutes)
        if current is None or start != current.ts_unix:
            if current is not None:
                yield current.to_candle()
            current = _OpenBucket(ts_unix=start, open=trade.price, high=trade.price, low=trade.price, close=trade.price)
        current.add(trade)
    if current is not None:
        yield current.to_candle()
