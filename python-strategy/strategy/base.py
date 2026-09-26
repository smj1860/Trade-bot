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
    ) -> Optional[StrategyDecision]:
        """Called once per order book update for a symbol this strategy
        trades. Returns None only when there isn't yet enough book state
        to compute features (e.g. a one-sided book right after connecting)
        — once features exist, always return a StrategyDecision, even
        when intent is None, so the engine can log what was considered."""
        raise NotImplementedError
