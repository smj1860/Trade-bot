"""
Buckets a stream of (timestamp, price) ticks into fixed-duration bars per
symbol, and keeps a rolling history of completed bars' close prices and
midpoint prices — the same "ordered list of bar values" shape
strategy/indicators.py's pure functions expect.

Bars here are built from mid-price ticks (there's no live trade/volume
feed wired into the strategy layer yet — see engine.py), not real traded
volume, unlike the OHLC candles historical-data/ pulls from Kraken's REST
API. That's an approximation, not something that breaks train/live
parity: the historical training script computes the exact same indicator
math (strategy.indicators) over Kraken's OHLC candle closes/highs/lows,
and this module produces the same shape (ordered lists of bar values)
from live ticks. What has to match between the two is the bar *interval*
(see strategy_config.example.toml's `bar_interval_minutes`, which should
agree with whatever interval_minutes the training data was pulled at) —
using the last tick in a bucket as its close (rather than a real traded
close), and the tick-range high/low as the bar's high/low (rather than
real traded highs/lows), are documented approximations, not silent ones;
see docs/model-training.md.

Bars are bucketed on event time, aligned to epoch (bucket start =
floor(timestamp / bar_interval_seconds) * bar_interval_seconds), not on a
fixed count of ticks — that's what makes a "bar" here mean the same thing
as an hourly/daily OHLC candle pulled from Kraken, rather than something
that speeds up or slows down with how chatty the book is.
"""

from __future__ import annotations

from collections import deque


class _SymbolBars:
    def __init__(self, max_bars: int) -> None:
        self.closes: deque[float] = deque(maxlen=max_bars)
        # (high + low) / 2 for each completed bar — the "typical/median
        # price" the Awesome Oscillator is traditionally computed on,
        # rather than the close.
        self.midpoints: deque[float] = deque(maxlen=max_bars)
        self.current_bucket_start: int | None = None
        self.current_close: float = 0.0
        self.current_high: float = 0.0
        self.current_low: float = 0.0


class BarAggregator:
    """Maintains, per symbol, a rolling history of completed bars (close
    and high/low-derived midpoint) built from a stream of (timestamp,
    price) ticks."""

    def __init__(self, bar_interval_seconds: int, max_bars: int) -> None:
        if bar_interval_seconds <= 0:
            raise ValueError("bar_interval_seconds must be positive")
        if max_bars <= 0:
            raise ValueError("max_bars must be positive")
        self._bar_interval_seconds = bar_interval_seconds
        self._max_bars = max_bars
        self._state: dict[str, _SymbolBars] = {}

    def on_tick(self, symbol: str, timestamp: float, price: float) -> None:
        """Feed one tick (e.g. mid-price at the current event time). When
        the tick falls into a new bucket, the previous bucket's last-seen
        price (its close) and its tick-range high/low midpoint are
        appended to that symbol's completed-bar history. The still-
        forming current bucket is never itself appended — callers only
        ever see completed bars, live or in training, so a model never
        has to handle a partially-formed final bar it wouldn't also see
        at inference time."""
        state = self._state.setdefault(symbol, _SymbolBars(self._max_bars))
        bucket_start = int(timestamp // self._bar_interval_seconds) * self._bar_interval_seconds

        if state.current_bucket_start is None:
            # First tick ever for this symbol: opens the first bucket, no
            # bar completes yet.
            state.current_bucket_start = bucket_start
            state.current_close = price
            state.current_high = price
            state.current_low = price
            return

        if bucket_start == state.current_bucket_start:
            state.current_close = price
            if price > state.current_high:
                state.current_high = price
            if price < state.current_low:
                state.current_low = price
            return

        if bucket_start < state.current_bucket_start:
            # An out-of-order/late tick (e.g. clock skew) — ignore rather
            # than rewrite already-completed history.
            return

        state.closes.append(state.current_close)
        state.midpoints.append((state.current_high + state.current_low) / 2.0)
        state.current_bucket_start = bucket_start
        state.current_close = price
        state.current_high = price
        state.current_low = price

    def closes(self, symbol: str) -> list[float]:
        """Completed bar closes for this symbol, oldest first."""
        state = self._state.get(symbol)
        if state is None:
            return []
        return list(state.closes)

    def midpoints(self, symbol: str) -> list[float]:
        """Completed bars' (high + low) / 2, oldest first — the "typical
        price" series the Awesome Oscillator is traditionally computed
        on."""
        state = self._state.get(symbol)
        if state is None:
            return []
        return list(state.midpoints)

    def window(self, symbol: str, size: int) -> list[float]:
        """The most recent `size` completed bar closes for this symbol
        (fewer if there isn't yet that much history) — the slice callers
        feed into strategy.indicators' close-based functions."""
        if size <= 0:
            return []
        return self.closes(symbol)[-size:]

    def midpoint_window(self, symbol: str, size: int) -> list[float]:
        """The most recent `size` completed bar midpoints for this symbol
        (fewer if there isn't yet that much history) — the slice callers
        feed into strategy.indicators' midpoint-based functions (e.g.
        awesome_oscillator)."""
        if size <= 0:
            return []
        return self.midpoints(symbol)[-size:]
