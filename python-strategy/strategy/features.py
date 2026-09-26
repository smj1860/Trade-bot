"""
Turns raw OrderBookUpdate events into a small, stable feature vector per
symbol. Strategies consume features from here, never raw book state
directly — that keeps strategy logic testable against canned feature
dicts instead of requiring a live socket.

Deliberately simple: this reads off the top of book (best bid/ask and
their quantities) rather than full depth. Order-book imbalance computed
from full depth is a reasonable improvement later, but the marginal value
over top-of-book is unclear without live results to justify the added
complexity — starting simple and auditable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Features:
    symbol: str
    mid_price: Decimal
    spread: Decimal
    imbalance: float  # in [-1, 1]; positive = more size on the bid
    momentum: float  # (mid_price - mid_price N updates ago) / mid_price N updates ago


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
    """Maintains rolling per-symbol state and computes a `Features` snapshot
    each time a new order book update arrives for that symbol."""

    def __init__(self, momentum_window: int) -> None:
        self._momentum_window = momentum_window
        self._state: dict[str, _SymbolState] = {}

    def on_order_book_update(
        self,
        symbol: str,
        best_bid_price: Decimal,
        best_bid_qty: Decimal,
        best_ask_price: Decimal,
        best_ask_qty: Decimal,
    ) -> Features | None:
        """Returns None if there isn't yet a usable two-sided book (e.g. a
        book that just connected and only has one side populated) — the
        caller should skip signal generation for that tick rather than act
        on a partial/degenerate book."""
        if best_bid_price <= 0 or best_ask_price <= 0 or best_ask_price <= best_bid_price:
            return None

        state = self._state.setdefault(symbol, _SymbolState(self._momentum_window))

        mid_price = (best_bid_price + best_ask_price) / 2
        spread = best_ask_price - best_bid_price

        total_qty = best_bid_qty + best_ask_qty
        imbalance = float((best_bid_qty - best_ask_qty) / total_qty) if total_qty > 0 else 0.0

        momentum = state.momentum(mid_price)

        state.mid_price_history.append(mid_price)

        return Features(
            symbol=symbol,
            mid_price=mid_price,
            spread=spread,
            imbalance=imbalance,
            momentum=momentum,
        )
