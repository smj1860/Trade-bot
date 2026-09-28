"""
Pure, dependency-free indicator math over an ordered list of bar closes
(oldest first). Deliberately kept as plain functions over plain lists,
with no notion of "live" or "historical" — the same functions are used
both by strategy/bars.py (fed one bar at a time as the live process runs)
and by historical-data's training script (fed a window sliced from a
pandas Series of historical closes). That's what guarantees a trained
model sees the exact same feature definition it'll be fed live: one
implementation, not two that have to be kept in sync by hand.

Every function returns 0.0 (a neutral value) when there isn't yet enough
history to compute a meaningful result, rather than raising or returning
None — callers (both live and training) can treat "not enough history
yet" as "no opinion" without a special case.
"""

from __future__ import annotations

import math
from typing import Sequence


def sma(closes: Sequence[float]) -> float:
    """Simple moving average of the given closes. Caller passes exactly
    the window it wants averaged (e.g. the last 20 bar closes)."""
    if not closes:
        return 0.0
    return sum(closes) / len(closes)


def sma_ratio(closes: Sequence[float]) -> float:
    """(most recent close / SMA of the whole window) - 1. Positive means
    price is above its recent average (a simple trend-following signal);
    negative means below. `closes` should be ordered oldest-first with
    the most recent close last."""
    if not closes:
        return 0.0
    average = sma(closes)
    if average == 0:
        return 0.0
    return (closes[-1] / average) - 1.0


def rsi(closes: Sequence[float]) -> float:
    """Standard Wilder-style RSI, scaled to [-1, 1] instead of the
    traditional [0, 100] so it composes with this project's other
    signals (0.5 avg gain/loss ratio maps to 0 in [-1, 1] rather than the
    traditional neutral 50 in [0, 100]). Needs at least 2 closes (1 price
    change) to produce a non-neutral result; returns 0.0 otherwise."""
    if len(closes) < 2:
        return 0.0
    gains = []
    losses = []
    for prev, curr in zip(closes, closes[1:]):
        change = curr - prev
        if change >= 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-change)
    avg_gain = sum(gains) / len(gains)
    avg_loss = sum(losses) / len(losses)
    if avg_gain == 0 and avg_loss == 0:
        return 0.0
    if avg_loss == 0:
        return 1.0
    rs = avg_gain / avg_loss
    traditional_rsi = 100.0 - (100.0 / (1.0 + rs))  # in [0, 100]
    return (traditional_rsi - 50.0) / 50.0  # rescale to [-1, 1]


def realized_vol(closes: Sequence[float]) -> float:
    """Standard deviation of log returns across the given closes — a
    simple realized-volatility estimate. Needs at least 3 closes (2 log
    returns) to compute a sample standard deviation; returns 0.0
    otherwise. Not scaled/annualized — the raw per-bar volatility, left
    for a model to calibrate against rather than assumed here."""
    if len(closes) < 3:
        return 0.0
    log_returns = []
    for prev, curr in zip(closes, closes[1:]):
        if prev <= 0 or curr <= 0:
            return 0.0
        log_returns.append(math.log(curr / prev))
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
    return math.sqrt(variance)


def bar_momentum(closes: Sequence[float]) -> float:
    """(most recent close - oldest close in the window) / oldest close —
    the bar-level equivalent of features.py's tick-level `momentum`, just
    computed over completed bars instead of raw ticks. Needs at least 2
    closes; returns 0.0 otherwise."""
    if len(closes) < 2 or closes[0] == 0:
        return 0.0
    return (closes[-1] - closes[0]) / closes[0]


def ema(closes: Sequence[float]) -> float:
    """Exponential moving average of the given closes, using the window's
    own length as the EMA period — mirrors sma()'s convention (the caller
    passes exactly the window it wants used) rather than taking a
    separate period argument. Standard smoothing factor alpha = 2/(N+1),
    seeded with the oldest close in the window. Needs at least 1 close;
    returns 0.0 otherwise."""
    if not closes:
        return 0.0
    alpha = 2.0 / (len(closes) + 1)
    value = closes[0]
    for c in closes[1:]:
        value = alpha * c + (1.0 - alpha) * value
    return value


def ema_ratio(closes: Sequence[float]) -> float:
    """(most recent close / EMA of the window) - 1 — the EMA-based analog
    of sma_ratio(): positive means price is above its EMA, negative means
    below. Needs at least 1 close; returns 0.0 otherwise."""
    if not closes:
        return 0.0
    average = ema(closes)
    if average == 0:
        return 0.0
    return (closes[-1] / average) - 1.0


def _stdev(values: Sequence[float]) -> float:
    """Population standard deviation (divides by N, not N-1) — matches
    the conventional Bollinger Bands formula, which uses the standard
    deviation of the same window the SMA is computed over. Needs at least
    2 values; returns 0.0 otherwise."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def bollinger_percent_b(closes: Sequence[float], num_std: float = 2.0) -> float:
    """Where the most recent close sits relative to its Bollinger Bands,
    rescaled from the traditional [0, 1] %b (0 = lower band, 1 = upper
    band) to roughly [-1, 1] (0 = middle band/SMA, +1 = upper band, -1 =
    lower band) so it composes with this project's other [-1, 1]-scaled
    signals. Deliberately left unclipped beyond +-1 — a close outside the
    bands (a real breakout) should look different from one sitting right
    at the band, not be capped to look the same. Needs at least 2 closes
    to compute a standard deviation, and a non-flat window (std > 0);
    returns 0.0 (neutral — at the average) otherwise."""
    if len(closes) < 2:
        return 0.0
    std = _stdev(closes)
    if std == 0:
        return 0.0
    middle = sma(closes)
    return (closes[-1] - middle) / (num_std * std)


def bollinger_bandwidth(closes: Sequence[float], num_std: float = 2.0) -> float:
    """Bollinger Band width relative to the middle band (SMA) — a
    volatility feature: wider bands (a bigger number) mean more recent
    price dispersion. Unlike realized_vol (std of log returns), this is
    std of raw price relative to price level, so it's on a similar scale
    across symbols at different price levels. Needs at least 2 closes and
    a non-zero average price; returns 0.0 otherwise."""
    if len(closes) < 2:
        return 0.0
    middle = sma(closes)
    if middle == 0:
        return 0.0
    std = _stdev(closes)
    return (2.0 * num_std * std) / middle


def _ema_series(values: Sequence[float], period: int) -> list[float]:
    """A full iterative EMA series (one value per input, not just the
    final value), using a *fixed* `period` argument — unlike ema(), which
    treats the whole window's length as the period. MACD needs this:
    its fast/slow EMAs use fixed periods (classically 12/26) regardless of
    how large a window of closes happens to be fed in, and the signal line
    is itself an EMA of the resulting MACD series, not of raw closes.
    Seeded with the first value, same convention as ema(). Returns an
    empty list if `values` is empty or `period` isn't positive."""
    if not values or period <= 0:
        return []
    alpha = 2.0 / (period + 1)
    series = [values[0]]
    for v in values[1:]:
        series.append(alpha * v + (1.0 - alpha) * series[-1])
    return series


def macd_histogram(
    closes: Sequence[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> float:
    """MACD histogram: (fast EMA - slow EMA) - signal-line EMA of that
    difference, i.e. how far the MACD line currently sits from its own
    signal line. Normalized by dividing by the most recent close (rather
    than left as a raw price-unit difference) so it's comparable across
    symbols at very different price levels, the same reasoning as
    bar_momentum's/sma_ratio's normalization. Classic default periods
    (12, 26, 9) per Gerald Appel's original definition. Needs at least
    `slow_period` closes to seed both EMAs, plus `signal_period` MACD
    values to seed the signal line; returns 0.0 otherwise, or if the most
    recent close is 0."""
    if len(closes) < slow_period or slow_period <= 0 or fast_period <= 0 or signal_period <= 0:
        return 0.0
    if closes[-1] == 0:
        return 0.0
    fast_series = _ema_series(closes, fast_period)
    slow_series = _ema_series(closes, slow_period)
    macd_series = [f - s for f, s in zip(fast_series, slow_series)]
    if len(macd_series) < signal_period:
        return 0.0
    signal_series = _ema_series(macd_series, signal_period)
    return (macd_series[-1] - signal_series[-1]) / closes[-1]


def cci(typical_prices: Sequence[float]) -> float:
    """Commodity Channel Index: how far the most recent typical price
    ((high + low + close) / 3 per bar — caller supplies this, since it
    needs high/low history, not just closes) sits from its SMA, relative
    to the mean absolute deviation of the window. Traditionally scaled so
    +-100 marks overbought/oversold; rescaled here by dividing by 100 to
    roughly match this project's [-1, 1] composability convention (the
    same idiom as rsi's/bollinger_percent_b's rescale), deliberately left
    unclipped beyond +-1 so a genuine extreme reading is still visible.
    Classic window is 20 typical prices. Needs at least 2 values to form
    a non-trivial average; returns 0.0 otherwise, or if the window is
    perfectly flat (mean absolute deviation of 0)."""
    if len(typical_prices) < 2:
        return 0.0
    average = sma(typical_prices)
    mean_abs_deviation = sum(abs(p - average) for p in typical_prices) / len(typical_prices)
    if mean_abs_deviation == 0:
        return 0.0
    raw = (typical_prices[-1] - average) / (0.015 * mean_abs_deviation)
    return raw / 100.0


def williams_percent_r(closes: Sequence[float], highs: Sequence[float], lows: Sequence[float]) -> float:
    """Classic Williams %R: where the most recent close sits within the
    window's high-low range, traditionally reported in [-100, 0] (0 = at
    the window's high, -100 = at the window's low). Rescaled here to
    roughly [-1, 1] (+1 = at the high, -1 = at the low, 0 = mid-range),
    the same rescale idiom as rsi's [-1, 1] mapping, so it composes with
    this project's other signals. `closes`, `highs`, and `lows` must be
    the same window (classic window is 14 bars), aligned bar-for-bar.
    Needs at least 1 bar of history and a non-flat range (highest high !=
    lowest low); returns 0.0 (neutral) otherwise, or on empty/mismatched
    input."""
    if not closes or not highs or not lows:
        return 0.0
    if len(closes) != len(highs) or len(closes) != len(lows):
        return 0.0
    highest_high = max(highs)
    lowest_low = min(lows)
    if highest_high == lowest_low:
        return 0.0
    raw = (highest_high - closes[-1]) / (highest_high - lowest_low) * -100.0
    return (raw + 50.0) / 50.0


def volume_ratio(volumes: Sequence[float]) -> float:
    """(most recent bar's traded volume / SMA of the window) - 1 — the
    same ratio-to-its-own-average idiom as sma_ratio()/ema_ratio(),
    applied to real traded volume (strategy/bars.py's `volumes` — 0.0 for
    a bar that only ever saw on_tick() book-snapshot ticks, never a real
    trade) instead of price. Positive means this bar traded more than its
    recent average (volume expansion — often accompanies a genuine
    breakout rather than noise); negative means below-average
    (exhaustion/quiet). Dimensionless by construction, exactly like
    sma_ratio(), so it's directly comparable across symbols with very
    different absolute volume — a low-cap altcoin's 10,000-unit bar and
    BTC's 500-unit bar can both read as "3x their own recent average."
    Needs at least one value; returns 0.0 otherwise, or when the window's
    average volume is 0 (e.g. no real trade feed wired up yet for this
    symbol — see BarAggregator.on_trade())."""
    if not volumes:
        return 0.0
    average = sma(volumes)
    if average == 0:
        return 0.0
    return (volumes[-1] / average) - 1.0


def parkinson_vol(highs: Sequence[float], lows: Sequence[float]) -> float:
    """Parkinson's high-low range volatility estimator: sqrt(mean(ln(high_i
    / low_i)^2) / (4 * ln 2)) over the window — a second, independent
    volatility read alongside realized_vol()'s close-to-close log-return
    standard deviation. Where realized_vol only sees where each bar
    *ended*, this sees how far price actually *traveled* intrabar (a bar
    that spiked hard in both directions before closing flat looks calm to
    realized_vol but clearly volatile here) — Parkinson (1980) showed this
    range-based estimator is markedly more statistically efficient than
    close-to-close for the same sample size, precisely because it uses
    information realized_vol discards. Like realized_vol, this is a ratio
    of prices (ln(high/low)), so it's already scale-free/dimensionless
    and directly comparable across symbols at very different price
    levels — no separate normalization needed. `highs` and `lows` must be
    the same window, aligned bar-for-bar. Needs at least 1 bar; returns
    0.0 otherwise, on mismatched lengths, or if any bar has a non-positive
    high or low (shouldn't happen with real price data)."""
    if not highs or not lows or len(highs) != len(lows):
        return 0.0
    squared_log_ranges = []
    for h, l in zip(highs, lows):
        if h <= 0 or l <= 0:
            return 0.0
        squared_log_ranges.append(math.log(h / l) ** 2)
    mean_squared = sum(squared_log_ranges) / len(squared_log_ranges)
    return math.sqrt(mean_squared / (4.0 * math.log(2.0)))


def awesome_oscillator(midpoints: Sequence[float], fast_window: int = 5, slow_window: int = 34) -> float:
    """Bill Williams' Awesome Oscillator: SMA(fast_window) of bar
    midpoints ((high + low) / 2, see strategy/bars.py) minus
    SMA(slow_window) of the same, expressed as a fraction of the slow SMA
    rather than a raw price-unit difference — that keeps it comparable
    across symbols at very different price levels (e.g. BTC vs. a
    low-priced altcoin), the same reasoning as bar_momentum's and
    sma_ratio's normalization. Classic default windows (5, 34) per
    Williams' original definition. Needs at least `slow_window` midpoints
    (the larger of the two SMAs); returns 0.0 otherwise."""
    if len(midpoints) < slow_window or slow_window <= 0 or fast_window <= 0:
        return 0.0
    fast = sma(midpoints[-fast_window:])
    slow = sma(midpoints[-slow_window:])
    if slow == 0:
        return 0.0
    return (fast - slow) / slow
