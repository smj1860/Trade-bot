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
- Bar-level (sma_ratio, ema_ratio, rsi, realized_vol, bar_momentum,
  bollinger_percent_b, bollinger_bandwidth, awesome_oscillator,
  macd_histogram, cci, williams_percent_r): computed by strategy.indicators
  over a rolling history of completed bars built by
  strategy.bars.BarAggregator from mid-price ticks. These are the ones a
  model can actually be trained on, because the exact same indicator
  functions run identically over Kraken's historical OHLC candle
  closes/highs/lows (see historical-data/'s training script) — see
  docs/model-training.md for why this split exists.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal

from strategy.bars import BarAggregator
from strategy.indicators import (
    atr_pct,
    awesome_oscillator,
    bar_momentum,
    bollinger_bandwidth,
    bollinger_percent_b,
    cci,
    ema_ratio,
    macd_histogram,
    parkinson_vol,
    realized_vol,
    returns_zscore,
    rsi,
    rsi_divergence,
    sma_ratio,
    subsample_tail,
    volume_ratio,
    vwap_ratio,
    williams_percent_r,
)


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
    ema_ratio: float = 0.0
    rsi: float = 0.0
    realized_vol: float = 0.0
    bar_momentum: float = 0.0
    bollinger_percent_b: float = 0.0
    bollinger_bandwidth: float = 0.0
    awesome_oscillator: float = 0.0
    macd_histogram: float = 0.0
    cci: float = 0.0
    williams_percent_r: float = 0.0
    # Volume-derived features (see strategy/indicators.py's volume_ratio/
    # parkinson_vol module docs). Both need strategy/bars.py's BarAggregator
    # to have real traded volume for this symbol (via on_trade()) to read
    # as anything but a flat 0.0/near-0.0 — a symbol only ever fed on_tick
    # book-snapshot ticks has no real volume history to compute these from,
    # same "no opinion" convention as every other not-enough-history case.
    volume_ratio: float = 0.0
    parkinson_vol: float = 0.0
    # Rolling Z-score of log returns (see strategy/indicators.py's
    # returns_zscore docstring for how this differs from
    # bollinger_percent_b (a price Z-score) and bar_momentum (a raw,
    # non-standardized cumulative return). Same neutral-default convention.
    returns_zscore: float = 0.0
    # Extended feature set (see scripts/train_model.py's EXTENDED_FEATURES;
    # same indicator functions, same window conventions as features_at()).
    ema_long_ratio: float = 0.0
    vwap_ratio: float = 0.0
    atr_pct: float = 0.0
    rsi_divergence: float = 0.0
    rsi_divergence_htf: float = 0.0


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
        ema_window: int = 12,
        rsi_window: int = 14,
        vol_window: int = 20,
        bar_momentum_window: int = 10,
        bollinger_window: int = 20,
        bollinger_num_std: float = 2.0,
        ao_fast_window: int = 5,
        ao_slow_window: int = 34,
        macd_fast_window: int = 12,
        macd_slow_window: int = 26,
        macd_signal_window: int = 9,
        cci_window: int = 20,
        williams_r_window: int = 14,
        ema_long_window: int = 200,
        vwap_window: int = 24,
        atr_window: int = 14,
        divergence_lookback: int = 14,
        htf_factor: int = 4,
    ) -> None:
        self._ema_long_window = ema_long_window
        self._vwap_window = vwap_window
        self._atr_window = atr_window
        self._divergence_lookback = divergence_lookback
        self._htf_factor = htf_factor
        self._momentum_window = momentum_window
        self._state: dict[str, _SymbolState] = {}

        self._sma_window = sma_window
        self._ema_window = ema_window
        self._rsi_window = rsi_window
        self._vol_window = vol_window
        self._bar_momentum_window = bar_momentum_window
        self._bollinger_window = bollinger_window
        self._bollinger_num_std = bollinger_num_std
        self._ao_fast_window = ao_fast_window
        self._ao_slow_window = ao_slow_window
        self._macd_fast_window = macd_fast_window
        self._macd_slow_window = macd_slow_window
        self._macd_signal_window = macd_signal_window
        self._cci_window = cci_window
        self._williams_r_window = williams_r_window
        # Every bar-derived indicator's required window, all folded into a
        # single max() — the underlying _SymbolBars deques (closes,
        # midpoints, highs, lows) all share one maxlen anyway, so there's
        # no benefit to sizing them separately. rsi/realized_vol need one
        # extra close to produce N price changes from a window of N+1
        # closes; MACD needs its slow EMA seeded plus enough MACD-series
        # history to seed the signal EMA on top of that.
        max_bars = (
            max(
                sma_window,
                ema_window,
                rsi_window + 1,
                vol_window + 1,
                bar_momentum_window,
                bollinger_window,
                ao_slow_window,
                macd_slow_window + macd_signal_window,
                cci_window,
                williams_r_window,
                ema_long_window,
                vwap_window,
                atr_window + 1,
                (rsi_window + divergence_lookback + 1) * htf_factor,
            )
            + 1
        )
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
            ema_ratio=ema_ratio(self._bars.window(symbol, self._ema_window)),
            rsi=rsi(self._bars.window(symbol, self._rsi_window + 1)),
            realized_vol=realized_vol(self._bars.window(symbol, self._vol_window + 1)),
            bar_momentum=bar_momentum(self._bars.window(symbol, self._bar_momentum_window)),
            bollinger_percent_b=bollinger_percent_b(
                self._bars.window(symbol, self._bollinger_window), self._bollinger_num_std
            ),
            bollinger_bandwidth=bollinger_bandwidth(
                self._bars.window(symbol, self._bollinger_window), self._bollinger_num_std
            ),
            awesome_oscillator=awesome_oscillator(
                self._bars.midpoint_window(symbol, self._ao_slow_window),
                self._ao_fast_window,
                self._ao_slow_window,
            ),
            macd_histogram=macd_histogram(
                self._bars.window(symbol, self._macd_slow_window + self._macd_signal_window),
                self._macd_fast_window,
                self._macd_slow_window,
                self._macd_signal_window,
            ),
            cci=cci(self._typical_price_window(symbol, self._cci_window)),
            williams_percent_r=williams_percent_r(
                self._bars.window(symbol, self._williams_r_window),
                self._bars.high_window(symbol, self._williams_r_window),
                self._bars.low_window(symbol, self._williams_r_window),
            ),
            volume_ratio=volume_ratio(self._bars.volume_window(symbol, self._vol_window)),
            parkinson_vol=parkinson_vol(
                self._bars.high_window(symbol, self._vol_window),
                self._bars.low_window(symbol, self._vol_window),
            ),
            returns_zscore=returns_zscore(self._bars.window(symbol, self._vol_window + 1)),
            ema_long_ratio=ema_ratio(self._bars.window(symbol, self._ema_long_window)),
            vwap_ratio=vwap_ratio(
                self._bars.window(symbol, self._vwap_window),
                self._bars.midpoint_window(symbol, self._vwap_window),
                self._bars.volume_window(symbol, self._vwap_window),
            ),
            atr_pct=atr_pct(
                self._bars.window(symbol, self._atr_window + 1),
                self._bars.high_window(symbol, self._atr_window + 1),
                self._bars.low_window(symbol, self._atr_window + 1),
            ),
            rsi_divergence=rsi_divergence(
                self._bars.window(symbol, self._rsi_window + self._divergence_lookback + 1),
                self._rsi_window,
                self._divergence_lookback,
            ),
            rsi_divergence_htf=rsi_divergence(
                subsample_tail(
                    self._bars.window(symbol, (self._rsi_window + self._divergence_lookback + 1) * self._htf_factor),
                    self._htf_factor,
                ),
                self._rsi_window,
                self._divergence_lookback,
            ),
        )

    def on_trade(
        self,
        symbol: str,
        price: Decimal,
        volume: Decimal,
        *,
        timestamp: float | None = None,
    ) -> None:
        """Feeds one real executed trade into this symbol's bars (see
        strategy/bars.py's BarAggregator.on_trade) — the source bar-derived
        features should actually be built from wherever a live trade feed
        is available, rather than only the mid-price-tick approximation
        on_order_book_update's own bar-feeding falls back to. Once a
        bucket has received any real trade, BarAggregator uses genuine
        VWAP for that bar's midpoint (see its module docstring), so this
        is what closes the live/historical feature-parity gap for symbols
        with a real trade feed wired up (see strategy/engine.py).

        Produces no Features snapshot: a trade print carries no bid/ask
        sizes to compute imbalance/momentum from, so there's nothing new
        to decide on here — the next order-book update for this symbol
        will see the updated bar history."""
        if timestamp is None:
            import time

            timestamp = time.time()
        self._bars.on_trade(symbol, timestamp, float(price), float(volume))

    def _typical_price_window(self, symbol: str, size: int) -> list[float]:
        """(high + low + close) / 3 per bar, over the most recent `size`
        completed bars — the "typical price" series CCI is traditionally
        computed on. Built here rather than in bars.py since it's a
        derived combination of three already-tracked histories, not a
        history BarAggregator needs to track itself."""
        closes = self._bars.window(symbol, size)
        highs = self._bars.high_window(symbol, size)
        lows = self._bars.low_window(symbol, size)
        return [(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)]
