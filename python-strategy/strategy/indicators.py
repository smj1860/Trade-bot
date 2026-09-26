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
