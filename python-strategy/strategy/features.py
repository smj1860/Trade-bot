"""
Turns raw OrderBookUpdate events into a small, stable feature vector per
symbol. Strategies consume features from here, never raw book state
directly — that keeps strategy logic testable against canned feature
dicts instead of requiring a live socket.

Two families of feature live here side by side:

- Tick-level (mid_price, spread, imbalance, momentum): computed directly
  off the top of book (best bid/ask and their quantities), unchanged from
  the original version of this module. These can't be reconstructed from
  historical OHLC candle data (imbalance needs live bid/ask sizes), so
  they're live-only — there's no historical training signal for them.
- Bar-level (sma_ratio, rsi, realized_vol, bar_momentum): computed by
  strategy.indicators over a rolling history of completed bars built by
  strategy.bars.BarAggregator from mid-price ticks. These are the ones a
  model can actually be trained on, because the exact same indicator
  functions run identically over Kraken's historical OHLC candle closes
  (see historical-data/'s training script) — see docs/model-training.md
  for why this split exists.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal

from strategy.bars import BarAggregator
from strategy.indicators import bar_momentum, realized_vol, rsi, sma_ratio


@dataclass(frozen=True)
class Features:
    symbol: str
    mid_price: Decimal
    spread: Decimal
    imbalance: float  # in [-1, 1]; positive = more size on the bid
    momentum: float  # (mid_price - mid_price N updates ago) / mid_price N updates ago
    # Bar-derived features (see module docstring). Default to 0.0 (a
    # neutral/"no opinion" value, matching strategy.indicators' own
    # not-enough-history convention) so existing callers that construct a
    # Features without these keeps working unchanged.
    sma_ratio: float = 0.0
    rsi: float = 0.0
    realized_vol: float = 0.0
    bar_momentum: float = 0.0


class _SymbolState:
    def __init__(self, momentum_window: int) -> None:
        self.mid_price_history: deque[Decimal] = deque(maxlen=momentum_window + 1)

    def momentum(self, current_mid: Decimal) -> float:
        if not self.mid_price_history:
            return 0.0
        reference = self.mid_price_history[0]
        if reference == 0:
            return 0.0
        return float((current_mid - reference) / reference)


class FeatureEngine:
    """Maintains rolling per-symbol state and computes a `Features`
    snapshot each time a new order book update arrives for that symbol."""

    def __init__(
        self,
        momentum_window: int,
        bar_interval_seconds: int = 3600,
        sma_window: int = 20,
        rsi_window: int = 14,
        vol_window: int = 20,
        bar_momentum_window: int = 10,
    ) -> None:
        self._momentum_window = momentum_window
        self._state: dict[str, _SymbolState] = {}

        self._sma_window = sma_window
        self._rsi_window = rsi_window
        self._vol_window = vol_window
        self._bar_momentum_window = bar_momentum_window
        # rsi/realized_vol need one extra close to produce N price changes
        # from a window of N+1 closes — size the bar history for the
        # largest window any of the four indicators actually needs.
        max_bars = max(sma_window, rsi_window + 1, vol_window + 1, bar_momentum_window) + 1
        self._bars = BarAggregator(bar_interval_seconds=bar_interval_seconds, max_bars=max_bars)

    def on_order_book_update(
        self,
        symbol: str,
        best_bid_price: Decimal,
        best_bid_qty: Decimal,
        best_ask_price: Decimal,
        best_ask_qty: Decimal,
        *,
        timestamp: float | None = None,
    ) -> Features | None:
        """Returns None if there isn't yet a usable two-sided book (e.g. a
        book that just connected and only has one side populated) — the
        caller should skip signal generation for that tick rather than act
        on a partial/degenerate book.

        `timestamp` is the event time (unix seconds) used to bucket ticks
        into bars for the bar-derived features below; it defaults to the
        current wall-clock time when the caller doesn't have a better one
        (e.g. in tests, or before engine.py threads through the real
        exchange timestamp)."""
        if best_bid_price <= 0 or best_ask_price <= 0 or best_ask_price <= best_bid_price:
            return None

        state = self._state.setdefault(symbol, _SymbolState(self._momentum_window))

        mid_price = (best_bid_price + best_ask_price) / 2
        spread = best_ask_price - best_bid_price

        total_qty = best_bid_qty + best_ask_qty
        imbalance = float((best_bid_qty - best_ask_qty) / total_qty) if total_qty > 0 else 0.0

        momentum = state.momentum(mid_price)

        state.mid_price_history.append(mid_price)

        if timestamp is None:
            import time

            timestamp = time.time()
        self._bars.on_tick(symbol, timestamp, float(mid_price))

        return Features(
            symbol=symbol,
            mid_price=mid_price,
            spread=spread,
            imbalance=imbalance,
            momentum=momentum,
            sma_ratio=sma_ratio(self._bars.window(symbol, self._sma_window)),
            rsi=rsi(self._bars.window(symbol, self._rsi_window + 1)),
            realized_vol=realized_vol(self._bars.window(symbol, self._vol_window + 1)),
            bar_momentum=bar_momentum(self._bars.window(symbol, self._bar_momentum_window)),
        )
