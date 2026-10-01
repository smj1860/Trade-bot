"""Pure logic for forward paper trading (scripts/paper_trade.py).

A paper trade mirrors one row of the triple-barrier backtest
(train_model.triple_barrier_net_pnl): enter at a closed bar's close, long or
short per the model, exit at whichever barrier is touched first by a later
bar's real high/low, or at the vertical barrier's close after `horizon`
bars. Three deliberate differences from the backtest, all in the direction
of honesty:

* Timeouts are KEPT. The backtest dataset drops rows that touch neither
  barrier in time (their label is unknowable in advance, so a live system
  cannot skip them); a paper trade is opened before anyone knows which
  outcome it will have, so every signal is scored.
* A same-bar double touch is not dropped. It is charged as a stop-out
  (the adverse barrier) and flagged ``ambiguous`` so the report can show how
  often it happens and how much it matters.
* Shorts are scored symmetrically, as in the backtest, but nothing here
  charges borrow/funding; ``holding_bars`` is recorded so the report can
  apply a borrow rate after the fact.

No I/O, no exchange or database code, so every rule is unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

LONG = 1
SHORT = -1


@dataclass(frozen=True)
class Bar:
    ts: int  # candle open time, unix seconds
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class Resolution:
    exit_reason: str  # "target" | "stop" | "timeout" | "ambiguous_stop"
    exit_ts: int
    exit_price: float
    holding_bars: int
    ambiguous: bool
    gross_return: float  # direction-adjusted, before costs
    net_return: float  # gross_return - round_trip_cost


def direction_from_prediction(predicted_up: int) -> int:
    return LONG if int(predicted_up) == 1 else SHORT


def resolve_trade(
    entry_price: float,
    direction: int,
    barrier_pct: float,
    horizon: int,
    round_trip_cost: float,
    bars_after_entry: Sequence[Bar],
) -> Optional[Resolution]:
    """Walks the bars that closed after the entry bar. Returns None while the
    trade is still open (no barrier touched and fewer than `horizon` bars
    available). `bars_after_entry` must be consecutive, closed bars in time
    order starting with the bar right after the entry bar."""
    if entry_price <= 0 or horizon < 1 or direction not in (LONG, SHORT):
        raise ValueError("entry_price > 0, horizon >= 1 and direction in (1, -1) required")
    upper = entry_price * (1.0 + barrier_pct)
    lower = entry_price * (1.0 - barrier_pct)
    for n, bar in enumerate(bars_after_entry[:horizon], start=1):
        hit_upper = bar.high >= upper
        hit_lower = bar.low <= lower
        if hit_upper and hit_lower:
            # Which came first intrabar is unknowable: charge the adverse
            # barrier for this direction (long loses at lower, short at upper).
            price = lower if direction == LONG else upper
            return _build(entry_price, direction, price, bar.ts, n, "ambiguous_stop", True, round_trip_cost)
        if hit_upper:
            reason = "target" if direction == LONG else "stop"
            return _build(entry_price, direction, upper, bar.ts, n, reason, False, round_trip_cost)
        if hit_lower:
            reason = "stop" if direction == LONG else "target"
            return _build(entry_price, direction, lower, bar.ts, n, reason, False, round_trip_cost)
    if len(bars_after_entry) >= horizon:
        last = bars_after_entry[horizon - 1]
        return _build(entry_price, direction, last.close, last.ts, horizon, "timeout", False, round_trip_cost)
    return None


def _build(
    entry: float,
    direction: int,
    exit_price: float,
    exit_ts: int,
    holding_bars: int,
    reason: str,
    ambiguous: bool,
    cost: float,
) -> Resolution:
    long_return = (exit_price - entry) / entry
    gross = direction * long_return
    return Resolution(reason, exit_ts, exit_price, holding_bars, ambiguous, gross, gross - cost)


def closed_bars(raw: Sequence[Sequence], interval_seconds: int, now: int) -> list[list]:
    """Drops Kraken's still-forming last candle: a candle is closed only once
    its open time + interval has passed."""
    return [c for c in raw if int(c[0]) + interval_seconds <= now]


def mean_and_ci(values: Sequence[float]) -> tuple[float, float, int]:
    """(mean, 95% half-width via normal approximation, n). Half-width is 0.0
    for n < 2 — there is no spread to estimate."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0, n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, 1.96 * (var / n) ** 0.5, n
