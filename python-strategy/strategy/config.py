"""
Loads strategy_config.toml (or whatever path STRATEGY_CONFIG_PATH points at)
into strongly-typed dataclasses. Mirrors the spirit of the Rust side's
config.rs: nothing about a specific asset or model is hardcoded, it's all
data from this file.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path


@dataclass(frozen=True)
class ConnectionConfig:
    rust_core_addr: str


@dataclass(frozen=True)
class FeatureConfig:
    momentum_window: int
    # Bar-derived feature parameters (see strategy/bars.py and
    # strategy/indicators.py). bar_interval_minutes should match whatever
    # interval_minutes the historical training data was pulled at (see
    # historical-data/'s ohlc_candles table and docs/model-training.md) —
    # that's what keeps a trained model's features identical to what the
    # live engine computes. Defaults below match the 60-minute candles
    # historical-data/backfill_ohlc.py backfills by default.
    bar_interval_minutes: int = 60
    sma_window: int = 20
    ema_window: int = 12
    rsi_window: int = 14
    vol_window: int = 20
    bar_momentum_window: int = 10
    bollinger_window: int = 20
    bollinger_num_std: float = 2.0
    # Bill Williams' classic Awesome Oscillator windows (fast/slow SMA of
    # bar midpoints). Kept separate from the other windows above since
    # AO's slow window (34) is much longer than anything else here.
    ao_fast_window: int = 5
    ao_slow_window: int = 34
    # MACD's classic fast/slow EMA periods plus the signal-line EMA period
    # applied to the MACD series itself.
    macd_fast_window: int = 12
    macd_slow_window: int = 26
    macd_signal_window: int = 9
    # CCI's classic typical-price window.
    cci_window: int = 20
    # Williams %R's classic high/low lookback window.
    williams_r_window: int = 14


@dataclass(frozen=True)
class ModelConfig:
    kind: str  # "rule_based" | "sklearn" | "torch"
    imbalance_weight: float
    momentum_weight: float
    model_path: str | None = None
    feature_order: tuple[str, ...] = ()


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    symbols: tuple[str, ...]
    exchange: str
    signal_threshold: float
    cooldown_seconds: float
    order_quantity: dict[str, Decimal]
    features: FeatureConfig
    model: ModelConfig


@dataclass(frozen=True)
class PortfolioConfig:
    max_position: dict[str, Decimal]


@dataclass(frozen=True)
class LoggingConfig:
    log_path: str
    level: str


@dataclass(frozen=True)
class ExecutionConfig:
    dry_run_only: bool = True


@dataclass(frozen=True)
class Config:
    connection: ConnectionConfig
    strategy: StrategyConfig
    portfolio: PortfolioConfig
    logging: LoggingConfig
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    @staticmethod
    def load(path: str | Path | None = None) -> "Config":
        if path is None:
            path = os.environ.get("STRATEGY_CONFIG_PATH", "strategy_config.example.toml")
        path = Path(path)
        with path.open("rb") as f:
            raw = tomllib.load(f)

        strategy_raw = raw["strategy"]
        model_raw = strategy_raw["model"]

        config = Config(
            connection=ConnectionConfig(**raw["connection"]),
            strategy=StrategyConfig(
                name=strategy_raw["name"],
                symbols=tuple(strategy_raw["symbols"]),
                exchange=strategy_raw["exchange"],
                signal_threshold=float(strategy_raw["signal_threshold"]),
                cooldown_seconds=float(strategy_raw["cooldown_seconds"]),
                order_quantity={k: Decimal(v) for k, v in strategy_raw["order_quantity"].items()},
                features=FeatureConfig(
                    momentum_window=int(strategy_raw["features"]["momentum_window"]),
                    bar_interval_minutes=int(strategy_raw["features"].get("bar_interval_minutes", 60)),
                    sma_window=int(strategy_raw["features"].get("sma_window", 20)),
                    ema_window=int(strategy_raw["features"].get("ema_window", 12)),
                    rsi_window=int(strategy_raw["features"].get("rsi_window", 14)),
                    vol_window=int(strategy_raw["features"].get("vol_window", 20)),
                    bar_momentum_window=int(strategy_raw["features"].get("bar_momentum_window", 10)),
                    bollinger_window=int(strategy_raw["features"].get("bollinger_window", 20)),
                    bollinger_num_std=float(strategy_raw["features"].get("bollinger_num_std", 2.0)),
                    ao_fast_window=int(strategy_raw["features"].get("ao_fast_window", 5)),
                    ao_slow_window=int(strategy_raw["features"].get("ao_slow_window", 34)),
                    macd_fast_window=int(strategy_raw["features"].get("macd_fast_window", 12)),
                    macd_slow_window=int(strategy_raw["features"].get("macd_slow_window", 26)),
                    macd_signal_window=int(strategy_raw["features"].get("macd_signal_window", 9)),
                    cci_window=int(strategy_raw["features"].get("cci_window", 20)),
                    williams_r_window=int(strategy_raw["features"].get("williams_r_window", 14)),
                ),
                model=ModelConfig(
                    kind=model_raw["kind"],
                    imbalance_weight=float(model_raw.get("imbalance_weight", 0.5)),
                    momentum_weight=float(model_raw.get("momentum_weight", 0.5)),
                    model_path=model_raw.get("model_path"),
                    feature_order=tuple(model_raw.get("feature_order", ())),
                ),
            ),
            portfolio=PortfolioConfig(
                max_position={k: Decimal(v) for k, v in raw["portfolio"]["max_position"].items()}
            ),
            logging=LoggingConfig(**raw["logging"]),
            execution=ExecutionConfig(**raw.get("execution", {})),
        )

        # Fail fast on an internally inconsistent config rather than
        # discovering it mid-run: every symbol the strategy trades needs a
        # configured order size, or we'll hit a KeyError deep inside a live
        # event loop instead of at startup.
        missing_sizes = [s for s in config.strategy.symbols if s not in config.strategy.order_quantity]
        if missing_sizes:
            raise ValueError(f"strategy.symbols includes symbols with no order_quantity configured: {missing_sizes}")

        return config
