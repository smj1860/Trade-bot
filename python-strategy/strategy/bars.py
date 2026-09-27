"""
Buckets a stream of live market events into fixed-duration bars per symbol,
and keeps a rolling history of completed bars' close prices and
midpoint/VWAP prices — the same "ordered list of bar values" shape
strategy/indicators.py's pure functions expect.

Two ways to feed a bar:

- on_trade(symbol, timestamp, price, volume): the preferred path, fed from
  Kraken's public trade tape (see kraken.rs's `trade` channel subscription
  and the TradeUpdate proto message) — real executed price and size. A
  bar built this way uses the real traded close/high/low and a genuine
  volume-weighted average price (VWAP = sum(price*volume) / sum(volume))
  for its "midpoint" series, matching what the OHLC candles
  historical-data/ pulls from Kraken's REST API actually represent
  (open/high/low/close/vwap/volume/trade_count), and what
  historical-data/ohlc_from_trades.py's aggregate_trades_to_candles()
  computes for historical backfills from the same trade tape — this
  module is the live-side twin of that aggregation.

- on_tick(symbol, timestamp, price): a fallback for mid-price/book
  snapshots, kept for symbols or periods where a real trade feed isn't
  available. A bar built entirely from ticks falls back to (high + low)/2
  as its "midpoint" (a documented approximation, not a silent one), since
  there's no real traded volume to weight by.

Both feed the same per-symbol bucket state, so a bucket that received at
least one real trade uses real VWAP for that bar even if it also saw
ticks; only a bucket with *zero* trades falls back to the tick-range
midpoint. In practice a symbol should be fed consistently by one path
(on_trade once the live engine's trade feed is wired up — see
strategy/features.py and strategy/engine.py), and mixing here is a
graceful-degradation seam, not the intended steady state.

Bars are bucketed on event time, aligned to epoch (bucket start =
floor(timestamp / bar_interval_seconds) * bar_interval_seconds), not on a
fixed count of ticks/trades — that's what makes a "bar" here mean the same
thing as an hourly/daily OHLC candle pulled from Kraken, rather than
something that speeds up or slows down with how chatty the feed is.

What has to match between live and historical training (see
docs/model-training.md) is the bar *interval* (strategy_config.example.toml's
`bar_interval_minutes` should agree with whatever interval_minutes the
training data was pulled at) — once on_trade is what's actually feeding a
symbol's bars, the live and historical feature computations are no longer
just interval-matched, they're built from the same *kind* of data (real
trades), closing the drift that mid-price-tick bars were a known,
documented approximation of.
"""

from __future__ import annotations

from collections import deque


class _SymbolBars:
    def __init__(self, max_bars: int) -> None:
        self.closes: deque[float] = deque(maxlen=max_bars)
        # VWAP (when the bucket saw at least one real trade) or (high +
        # low) / 2 (a tick-only bucket's fallback) for each completed bar —
        # the "typical price" series the Awesome Oscillator is
        # traditionally computed on, rather than the close. See
        # _complete_bucket().
        self.midpoints: deque[float] = deque(maxlen=max_bars)
        # Raw per-bar high/low history — needed by indicators that can't be
        # derived from the midpoint alone: CCI's typical price
        # ((high+low+close)/3) and Williams %R's highest-high/lowest-low
        # over a window both need the actual high and low, not just their
        # average.
        self.highs: deque[float] = deque(maxlen=max_bars)
        self.lows: deque[float] = deque(maxlen=max_bars)
        # Real traded volume per completed bar — 0.0 for a bar built
        # entirely from on_tick() calls (no real trade ever reported a
        # size for it).
        self.volumes: deque[float] = deque(maxlen=max_bars)
        self.current_bucket_start: int | None = None
        self.current_close: float = 0.0
        self.current_high: float = 0.0
        self.current_low: float = 0.0
        self.current_volume: float = 0.0
        self.current_pv_sum: float = 0.0  # sum of price * volume, for this bucket's VWAP


class BarAggregator:
    """Maintains, per symbol, a rolling history of completed bars (close,
    high/low, VWAP-or-midpoint, and traded volume) built from a stream of
    real trades (on_trade, preferred) and/or mid-price ticks (on_tick,
    fallback) — see the module docstring."""

    def __init__(self, bar_interval_seconds: int, max_bars: int) -> None:
        if bar_interval_seconds <= 0:
            raise ValueError("bar_interval_seconds must be positive")
        if max_bars <= 0:
            raise ValueError("max_bars must be positive")
        self._bar_interval_seconds = bar_interval_seconds
        self._max_bars = max_bars
        self._state: dict[str, _SymbolBars] = {}

    def _bucket_start(self, timestamp: float) -> int:
        return int(timestamp // self._bar_interval_seconds) * self._bar_interval_seconds

    def _complete_bucket(self, state: _SymbolBars) -> None:
        """Appends the just-finished bucket to this symbol's completed-bar
        history. VWAP is used for the midpoint series whenever the bucket
        accumulated any real traded volume (via on_trade); a bucket built
        entirely from on_tick calls has current_volume == 0.0 and falls
        back to (high + low) / 2, the same approximation this module has
        always used for a tick-only feed."""
        state.closes.append(state.current_close)
        if state.current_volume:
            vwap = state.current_pv_sum / state.current_volume
        else:
            vwap = (state.current_high + state.current_low) / 2.0
        state.midpoints.append(vwap)
        state.highs.append(state.current_high)
        state.lows.append(state.current_low)
        state.volumes.append(state.current_volume)

    def on_tick(self, symbol: str, timestamp: float, price: float) -> None:
        """Feed one mid-price/book-snapshot tick — the fallback path (see
        module docstring) for a symbol without a live trade feed wired up.
        When the tick falls into a new bucket, the previous bucket is
        completed (see _complete_bucket) and appended to that symbol's
        history. The still-forming current bucket is never itself
        appended — callers only ever see completed bars, live or in
        training, so a model never has to handle a partially-formed final
        bar it wouldn't also see at inference time."""
        state = self._state.setdefault(symbol, _SymbolBars(self._max_bars))
        bucket_start = self._bucket_start(timestamp)

        if state.current_bucket_start is None:
            # First event ever for this symbol: opens the first bucket, no
            # bar completes yet.
            state.current_bucket_start = bucket_start
            state.current_close = price
            state.current_high = price
            state.current_low = price
            state.current_volume = 0.0
            state.current_pv_sum = 0.0
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

        self._complete_bucket(state)
        state.current_bucket_start = bucket_start
        state.current_close = price
        state.current_high = price
        state.current_low = price
        state.current_volume = 0.0
        state.current_pv_sum = 0.0

    def on_trade(self, symbol: str, timestamp: float, price: float, volume: float) -> None:
        """Feed one real executed trade — the preferred path (see module
        docstring): real traded price and size, from Kraken's public trade
        tape. Bucketing and bar-completion rules are identical to on_tick
        (event-time bucketing, only completed buckets are ever exposed),
        but each trade also accumulates real volume and a price*volume sum
        so the completed bar's midpoint series is a genuine VWAP rather
        than the (high+low)/2 approximation a tick-only feed falls back
        to. `volume` should be > 0 for a real trade; a non-positive volume
        is accepted (it still moves the close/high/low) but contributes
        nothing to the VWAP weighting, same as it wouldn't in a real
        volume-weighted average."""
        state = self._state.setdefault(symbol, _SymbolBars(self._max_bars))
        bucket_start = self._bucket_start(timestamp)

        if state.current_bucket_start is None:
            state.current_bucket_start = bucket_start
            state.current_close = price
            state.current_high = price
            state.current_low = price
            state.current_volume = volume
            state.current_pv_sum = price * volume
            return

        if bucket_start == state.current_bucket_start:
            state.current_close = price
            if price > state.current_high:
                state.current_high = price
            if price < state.current_low:
                state.current_low = price
            state.current_volume += volume
            state.current_pv_sum += price * volume
            return

        if bucket_start < state.current_bucket_start:
            # An out-of-order/late trade (e.g. a reordered feed message) —
            # ignore rather than rewrite already-completed history, same
            # as on_tick.
            return

        self._complete_bucket(state)
        state.current_bucket_start = bucket_start
        state.current_close = price
        state.current_high = price
        state.current_low = price
        state.current_volume = volume
        state.current_pv_sum = price * volume

    def closes(self, symbol: str) -> list[float]:
        """Completed bar closes for this symbol, oldest first."""
        state = self._state.get(symbol)
        if state is None:
            return []
        return list(state.closes)

    def midpoints(self, symbol: str) -> list[float]:
        """Completed bars' VWAP (bars fed via on_trade) or (high + low) / 2
        (bars fed only via on_tick), oldest first — the "typical price"
        series the Awesome Oscillator is traditionally computed on. See
        the module docstring for which one a given bar used."""
        state = self._state.get(symbol)
        if state is None:
            return []
        return list(state.midpoints)

    def highs(self, symbol: str) -> list[float]:
        """Completed bars' highs, oldest first."""
        state = self._state.get(symbol)
        if state is None:
            return []
        return list(state.highs)

    def lows(self, symbol: str) -> list[float]:
        """Completed bars' lows, oldest first."""
        state = self._state.get(symbol)
        if state is None:
            return []
        return list(state.lows)

    def volumes(self, symbol: str) -> list[float]:
        """Completed bars' real traded volume (from on_trade calls only —
        0.0 for a bar built entirely from on_tick), oldest first."""
        state = self._state.get(symbol)
        if state is None:
            return []
        return list(state.volumes)

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

    def high_window(self, symbol: str, size: int) -> list[float]:
        """The most recent `size` completed bar highs for this symbol
        (fewer if there isn't yet that much history)."""
        if size <= 0:
            return []
        return self.highs(symbol)[-size:]

    def low_window(self, symbol: str, size: int) -> list[float]:
        """The most recent `size` completed bar lows for this symbol
        (fewer if there isn't yet that much history)."""
        if size <= 0:
            return []
        return self.lows(symbol)[-size:]

    def volume_window(self, symbol: str, size: int) -> list[float]:
        """The most recent `size` completed bars' real traded volume for
        this symbol (fewer if there isn't yet that much history)."""
        if size <= 0:
            return []
        return self.volumes(symbol)[-size:]
