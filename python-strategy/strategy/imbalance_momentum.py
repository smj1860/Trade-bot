"""
The first concrete Strategy: combines order-book imbalance and short-term
momentum (via RuleBasedModel, see models.py) into a signal, then a
DecisionPolicy turns that into an order or a no-op. Genuinely runnable
against live Kraken data today — not a placeholder — but a simple rule,
not a trained model. See models.py's module docstring for why that's the
honest starting point here.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from strategy.base import Strategy, StrategyDecision
from strategy.config import Config
from strategy.features import FeatureEngine
from strategy.models import ModelWrapper, build_model
from strategy.policy import DecisionPolicy


class ImbalanceMomentumStrategy(Strategy):
    def __init__(self, config: Config, strategy_id: str = "imbalance-momentum-v1") -> None:
        self.strategy_id = strategy_id
        self._tradeable_symbols = set(config.strategy.symbols)
        self._features = FeatureEngine(
            momentum_window=config.strategy.features.momentum_window,
            bar_interval_seconds=config.strategy.features.bar_interval_minutes * 60,
            sma_window=config.strategy.features.sma_window,
            ema_window=config.strategy.features.ema_window,
            rsi_window=config.strategy.features.rsi_window,
            vol_window=config.strategy.features.vol_window,
            bar_momentum_window=config.strategy.features.bar_momentum_window,
            bollinger_window=config.strategy.features.bollinger_window,
            bollinger_num_std=config.strategy.features.bollinger_num_std,
            ao_fast_window=config.strategy.features.ao_fast_window,
            ao_slow_window=config.strategy.features.ao_slow_window,
            macd_fast_window=config.strategy.features.macd_fast_window,
            macd_slow_window=config.strategy.features.macd_slow_window,
            macd_signal_window=config.strategy.features.macd_signal_window,
            cci_window=config.strategy.features.cci_window,
            williams_r_window=config.strategy.features.williams_r_window,
        )
        self._model: ModelWrapper = build_model(
            config.strategy.model.kind,
            imbalance_weight=config.strategy.model.imbalance_weight,
            momentum_weight=config.strategy.model.momentum_weight,
            model_path=config.strategy.model.model_path,
            feature_order=config.strategy.model.feature_order,
        )
        self._policy = DecisionPolicy(
            signal_threshold=config.strategy.signal_threshold,
            cooldown_seconds=config.strategy.cooldown_seconds,
            order_quantity=config.strategy.order_quantity,
            max_position=config.portfolio.max_position,
        )

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
        if symbol not in self._tradeable_symbols:
            return None

        features = self._features.on_order_book_update(
            symbol, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty, timestamp=timestamp
        )
        if features is None:
            return None

        signal = self._model.predict(features)
        intent = self._policy.decide(features, signal, current_position)
        return StrategyDecision(features=features, signal=signal, intent=intent)
