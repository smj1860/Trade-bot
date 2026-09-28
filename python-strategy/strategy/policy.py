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

Institutional audit Phase 1.3: orders built here are LIMIT/post-only by
default (see `use_limit_orders`), priced to rest at the current best
bid (BUY) / best ask (SELL) rather than cross the book — a market order's
~0.8% one-way taker fee against a signal that hasn't cleared coin-flip
accuracy is a guaranteed loser regardless of model quality (see the
institutional audit). `use_limit_orders=False` restores the original
always-MARKET behavior. The actual place -> wait -> cancel/reprice ->
fallback-to-market lifecycle for a resting order lives in
strategy/engine.py's `_manage_resting_order`, not here — this layer only
decides *what* to submit first, not how to manage it afterward.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Optional

from strategy.config import SizingConfig
from strategy.features import Features

Side = Literal["BUY", "SELL"]
OrderType = Literal["MARKET", "LIMIT"]


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: Side
    quantity: Decimal
    signal: float  # carried through purely for logging/audit, not re-used downstream
    order_type: OrderType = "MARKET"
    # Set iff order_type == "LIMIT": the post-only price this order should
    # rest at (current best bid for a BUY, current best ask for a SELL) —
    # see this module's docstring and engine.py's _manage_resting_order,
    # which re-derives a fresh price at each reprice attempt rather than
    # reusing this one, since the book has likely moved by then. This
    # field is only ever the *initial* price.
    limit_price: Optional[Decimal] = None


class DecisionPolicy:
    def __init__(
        self,
        *,
        signal_threshold: float,
        cooldown_seconds: float,
        order_quantity: dict[str, Decimal],
        max_position: dict[str, Decimal],
        use_limit_orders: bool = True,
        sizing: SizingConfig | None = None,
    ) -> None:
        self._signal_threshold = signal_threshold
        self._cooldown_seconds = cooldown_seconds
        self._order_quantity = order_quantity
        self._max_position = max_position
        self._use_limit_orders = use_limit_orders
        self._sizing = sizing if sizing is not None else SizingConfig()
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
        base_quantity = self._order_quantity[features.symbol]
        quantity = scaled_quantity(
            base_quantity, signal, self._signal_threshold, features.realized_vol, self._sizing
        )
        signed_qty = quantity if side == "BUY" else -quantity
        projected_position = current_position + signed_qty

        max_pos = self._max_position.get(features.symbol)
        if max_pos is not None and abs(projected_position) > max_pos:
            return None

        self._last_order_time[features.symbol] = now

        if not self._use_limit_orders:
            return OrderIntent(symbol=features.symbol, side=side, quantity=quantity, signal=signal)

        limit_price = post_only_price(side, features.mid_price, features.spread)
        return OrderIntent(
            symbol=features.symbol,
            side=side,
            quantity=quantity,
            signal=signal,
            order_type="LIMIT",
            limit_price=limit_price,
        )


def scaled_quantity(
    base_quantity: Decimal,
    signal: float,
    signal_threshold: float,
    realized_vol: float,
    sizing: SizingConfig,
) -> Decimal:
    """Institutional audit Phase 2.1: scales `base_quantity` (the flat
    per-symbol size from strategy.order_quantity) by inverse recent
    volatility and signal conviction, instead of submitting the identical
    size on every trade regardless of current market conditions or
    signal strength.

    Returns `base_quantity` UNCHANGED (the flat fallback the audit asked
    for) when:
      - `sizing.enabled` is False, or
      - `realized_vol <= 0` — either genuinely calm-to-the-point-of-zero
        (vanishingly rare with a fee-clearing move label) or, far more
        commonly, a symbol that hasn't accumulated enough bar history yet
        for FeatureEngine to report anything but its neutral 0.0 "no
        opinion" default (see features.py) — treating that as "target
        volatility exactly met" would be a silent, wrong assumption, not
        a safe default.

    Otherwise: `base_quantity * volatility_scalar * conviction_scalar`,
    where:
      volatility_scalar = clamp(target_volatility / realized_vol,
                                 min_size_multiplier, max_size_multiplier)
        — a symbol currently calmer than the target gets sized up (more
        size for the same implied dollar-risk budget); a symbol currently
        more volatile than the target gets sized down.
      conviction_scalar = min_conviction_multiplier + (1 -
                           min_conviction_multiplier) * conviction_fraction
        — linear from min_conviction_multiplier at |signal| ==
        signal_threshold (the weakest signal that clears the trade gate
        at all) up to 1.0 at |signal| == 1.0 (the strongest possible
        signal). `decide()` already guarantees |signal| >=
        signal_threshold by the time this is called, and
        conviction_fraction is clamped to [0, 1] regardless as a
        defensive measure against an out-of-range caller."""
    if not sizing.enabled or realized_vol <= 0:
        return base_quantity

    volatility_scalar = sizing.target_volatility / realized_vol
    volatility_scalar = max(sizing.min_size_multiplier, min(sizing.max_size_multiplier, volatility_scalar))

    if signal_threshold < 1.0:
        conviction_fraction = (abs(signal) - signal_threshold) / (1.0 - signal_threshold)
        conviction_fraction = max(0.0, min(1.0, conviction_fraction))
    else:
        conviction_fraction = 1.0
    conviction_scalar = sizing.min_conviction_multiplier + (1.0 - sizing.min_conviction_multiplier) * conviction_fraction

    combined = volatility_scalar * conviction_scalar
    combined = max(sizing.min_size_multiplier, min(sizing.max_size_multiplier, combined))
    return base_quantity * Decimal(str(combined))


def post_only_price(side: Side, mid_price: Decimal, spread: Decimal) -> Decimal:
    """The post-only (maker) price to rest at: the current best bid for a
    BUY, the current best ask for a SELL. `Features` carries `mid_price`
    and `spread` rather than raw best bid/ask (see features.py), but since
    `spread = best_ask - best_bid` and `mid_price = (best_bid + best_ask)
    / 2`, both are exact inverses of those two — `best_bid = mid_price -
    spread / 2`, `best_ask = mid_price + spread / 2` — so nothing is lost
    recomputing them here rather than threading raw book levels through
    Features just for this. Joining the best price (rather than improving
    on it by a tick) is deliberately the simplest correct choice for a
    first version: it's guaranteed non-crossing (see risk.rs's
    check_post_only_would_not_cross) without needing this layer to also
    know each symbol's tick size."""
    half_spread = spread / 2
    return mid_price - half_spread if side == "BUY" else mid_price + half_spread
