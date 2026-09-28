"""
Tracks net position per symbol, client-side, so the strategy/policy layer
can make position-aware decisions (see policy.py's soft max-position
check) without querying Rust for it on every tick.

Important, honest limitation: this only updates on a CONFIRMED fill
(status FILLED or PARTIALLY_FILLED with a filled_quantity), never on
ORDER_STATUS_ACCEPTED. That's deliberate — Kraken accepting an order is
not the same as it being filled, and inferring a fill from acceptance
would be a guess dressed up as data.

The Rust core's `StreamOrderUpdates` RPC streams real fills from Kraken's
private WebSocket feed (see rust-core/src/kraken_private_ws.rs and
order.rs) — this is not a stub, and engine.py's `_order_update_loop`
applies every update it receives here.

`filled_quantity` on an `OrderUpdate` is CUMULATIVE for that order (it's
sourced from Kraken's own `cum_qty` — see kraken_private_ws.rs's
`build_status_update`), not a per-update delta. A single order can
legitimately produce more than one update over its lifetime — a partial
fill, then another partial, then a final fill — especially now that
institutional audit Phase 1.3's resting limit orders can sit on the book
across several ticks. Applying each update's `filled_quantity` directly
to the position (as an early version of this class did) double- and
triple-counts every fill after the first for the same order.
`on_order_update` tracks the last-applied cumulative amount per
`client_order_id` and only applies the incremental delta, so repeated
updates for one order accumulate correctly regardless of how many arrive
or how far apart.

This is purely advisory bookkeeping for the Python strategy layer. It is
never the source of truth for real risk limits — that's the Rust risk
engine, which tracks its own (equally zero-until-fills-exist) positions
independently and is the one that actually blocks orders.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from strategy.policy import Side

# Statuses that represent a confirmed change in actual holdings. Matches
# the OrderStatus enum names in trading.proto with the ORDER_STATUS_
# prefix stripped, since callers translate from the proto enum before
# calling in — this module has no dependency on generated protobuf code,
# which keeps it trivially unit-testable.
_FILL_STATUSES = frozenset({"FILLED", "PARTIALLY_FILLED"})


class PortfolioManager:
    def __init__(self) -> None:
        self._positions: dict[str, Decimal] = {}
        # client_order_id -> cumulative filled_quantity last applied for
        # that order, so a later update for the same order only applies
        # the new incremental amount (see this module's docstring).
        self._cumulative_filled: dict[str, Decimal] = {}

    def position(self, symbol: str) -> Decimal:
        return self._positions.get(symbol, Decimal(0))

    def on_order_update(
        self,
        order_id: str,
        symbol: str,
        side: Side,
        status: str,
        filled_quantity: Optional[Decimal],
    ) -> None:
        if status not in _FILL_STATUSES or filled_quantity is None or filled_quantity == 0:
            return
        previously_applied = self._cumulative_filled.get(order_id, Decimal(0))
        delta = filled_quantity - previously_applied
        if delta <= 0:
            # A duplicate, stale, or out-of-order update carrying no new
            # fill beyond what's already been applied for this order —
            # nothing to do. (A negative delta would mean cum_qty went
            # backwards, which shouldn't happen; treated the same as "no
            # new fill" rather than as a reason to subtract from the
            # position.)
            return
        self._cumulative_filled[order_id] = filled_quantity
        signed = delta if side == "BUY" else -delta
        self._positions[symbol] = self.position(symbol) + signed

    def snapshot(self) -> dict[str, Decimal]:
        """Non-zero positions only, for logging/inspection."""
        return {symbol: qty for symbol, qty in self._positions.items() if qty != 0}
