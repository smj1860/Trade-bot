"""
Strategy interface. Every concrete strategy — rule-based today, ML-driven
later — implements this same contract, so the engine (engine.py) never
needs to know which one it's driving. Deliberately proto-free: a strategy
takes and returns plain Python/Decimal types, translated to/from the gRPC
wire format only inside engine.py. That keeps strategies unit-testable
without a running Rust server or generated protobuf code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from strategy.features import Features
from strategy.policy import OrderIntent


@dataclass(frozen=True)
class StrategyDecision:
    """Everything worth logging about one tick's worth of thinking, not
    just the outcome — `intent` is often None (no trade), and that's a
    normal, loggable result, not an absence of one."""

    features: Features
    signal: float
    intent: Optional[OrderIntent]


class Strategy(ABC):
    strategy_id: str

    @abstractmethod
    def on_order_book_update(
        self,
        symbol: str,
        best_bid_price: Decimal,
        best_bid_qty: Decimal,
        best_ask_price: Decimal,
        best_ask_qty: Decimal,
        current_position: Decimal,
        *,
        timestamp: Optional[float] = None,
    ) -> Optional[StrategyDecision]:
        """Called once per order book update for a symbol this strategy
        trades. Returns None only when there isn't yet enough book state
        to compute features (e.g. a one-sided book right after connecting)
        — once features exist, always return a StrategyDecision, even
        when intent is None, so the engine can log what was considered.

        `timestamp` is the event's unix-seconds time, used by bar-derived
        features (see strategy/bars.py); optional and keyword-only so
        existing callers that don't have a timestamp handy (tests, mainly)
        keep working unchanged — the feature engine falls back to
        wall-clock time when it's omitted."""
        raise NotImplementedError

    def on_trade(
        self,
        symbol: str,
        price: Decimal,
        volume: Decimal,
        *,
        timestamp: Optional[float] = None,
    ) -> None:
        """Called once per real executed trade for a symbol this strategy
        trades (see proto/trading.proto's TradeUpdate and
        strategy/engine.py's market-data loop) — feeds bar-derived
        features real traded price/volume instead of the mid-price-tick
        approximation on_order_book_update falls back to. Unlike that
        method, this never produces a StrategyDecision on its own: a trade
        print carries no bid/ask sizes to compute imbalance/momentum from,
        so there's nothing to decide on here — a bar completing from this
        call still shapes the *next* order-book-driven decision's
        bar-derived features.

        Default no-op, so a strategy with no bar-derived features (or a
        caller/test with no trade feed wired up) needs no override.
        ImbalanceMomentumStrategy overrides this to feed its
        FeatureEngine."""
        return None
