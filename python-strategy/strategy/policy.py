"""
Converts a model's signal into a concrete order (or a decision not to
trade). This is the layer that owns "what should we do about it" — kept
separate from the model (strategy/models.py, which only answers "what do
we think") so sizing/thresholds stay visible and testable independent of
whatever produced the signal.

Two independent gates before an order gets built, both config-driven and
both logged when they block a trade:
  1. Cooldown — don't re-fire on the same symbol faster than
     `cooldown_seconds`, regardless of signal.
  2. Soft position limit — don't add to a position that's already at or
     past its configured advisory max in that direction. This is a
     courtesy to cut down on orders that Rust's real risk engine would
     reject anyway; it is NOT a substitute for that risk engine, which
     stays the actual enforcement point.

Orders built here are always MARKET orders. A first version has to pick
something, and market orders let the Rust risk engine price against the
live book (already tested) rather than requiring this layer to also
decide a sensible limit price — a real limit-order strategy is a
reasonable future addition, not a hard requirement for a working v1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Optional

from strategy.features import Features

Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: Side
    quantity: Decimal
    signal: float  # carried through purely for logging/audit, not re-used downstream


class DecisionPolicy:
    def __init__(
        self,
        *,
        signal_threshold: float,
        cooldown_seconds: float,
        order_quantity: dict[str, Decimal],
        max_position: dict[str, Decimal],
    ) -> None:
        self._signal_threshold = signal_threshold
        self._cooldown_seconds = cooldown_seconds
        self._order_quantity = order_quantity
        self._max_position = max_position
        self._last_order_time: dict[str, float] = {}

    def decide(
        self,
        features: Features,
        signal: float,
        current_position: Decimal,
        now: Optional[float] = None,
    ) -> Optional[OrderIntent]:
        now = time.monotonic() if now is None else now

        if abs(signal) < self._signal_threshold:
            return None

        last = self._last_order_time.get(features.symbol)
        if last is not None and (now - last) < self._cooldown_seconds:
            return None

        side: Side = "BUY" if signal > 0 else "SELL"
        quantity = self._order_quantity[features.symbol]
        signed_qty = quantity if side == "BUY" else -quantity
        projected_position = current_position + signed_qty

        max_pos = self._max_position.get(features.symbol)
        if max_pos is not None and abs(projected_position) > max_pos:
            return None

        self._last_order_time[features.symbol] = now
        return OrderIntent(symbol=features.symbol, side=side, quantity=quantity, signal=signal)
