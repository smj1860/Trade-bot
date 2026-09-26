"""
Tracks net position per symbol, client-side, so the strategy/policy layer
can make position-aware decisions (see policy.py's soft max-position
check) without querying Rust for it on every tick.

Important, honest limitation: this only updates on a CONFIRMED fill
(status FILLED or PARTIALLY_FILLED with a filled_quantity), never on
ORDER_STATUS_ACCEPTED. That's deliberate — Kraken accepting an order is
not the same as it being filled, and inferring a fill from acceptance
would be a guess dressed up as data.

As of this writing, that means this will rarely update in practice: the
Rust core's `StreamOrderUpdates` RPC (the only place a real fill
notification could come from) is still an empty stream stub
(rust-core/src/order.rs) — there is no fill-tracking pipeline from Kraken
back into Rust yet, let alone from Rust into Python. This class is built
and tested against the data shape it will need once that exists, not
validated against real fills, because there aren't any to validate
against yet. Don't read "PortfolioManager exists" as "position tracking
works end-to-end" — it doesn't, yet.

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

    def position(self, symbol: str) -> Decimal:
        return self._positions.get(symbol, Decimal(0))

    def on_order_update(
        self,
        symbol: str,
        side: Side,
        status: str,
        filled_quantity: Optional[Decimal],
    ) -> None:
        if status not in _FILL_STATUSES or filled_quantity is None or filled_quantity == 0:
            return
        signed = filled_quantity if side == "BUY" else -filled_quantity
        self._positions[symbol] = self.position(symbol) + signed

    def snapshot(self) -> dict[str, Decimal]:
        """Non-zero positions only, for logging/inspection."""
        return {symbol: qty for symbol, qty in self._positions.items() if qty != 0}
