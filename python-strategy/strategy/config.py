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
    # Institutional audit Phase 1.1's dead-man's switch: how often this
    # process calls OrderService.SendHeartbeat. Should be comfortably under
    # the Rust side's configured dead_man_switch.heartbeat_timeout_secs
    # (default 30s there) — the default here (5s) gives six heartbeats per
    # timeout window, tolerant of an occasional missed/slow call without
    # tripping the switch. See rust-core/src/heartbeat.rs.
    heartbeat_interval_secs: float = 5.0


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
class SizingConfig:
    """Institutional audit Phase 2.1: scales strategy.order_quantity's flat
    per-symbol base size by inverse recent volatility (vol-targeting) and
    signal conviction, instead of submitting the identical size on every
    trade regardless of current market conditions or how strong the signal
    is. See strategy/policy.py's scaled_quantity().

    `order_quantity[symbol]` remains the reference size this scales
    relative to, and the fallback used verbatim whenever scaling can't be
    computed (enabled=False, or a symbol without enough bar history yet
    for a real realized_vol reading — see Features.realized_vol's
    "no opinion" 0.0 default)."""

    enabled: bool = True
    # The realized_vol level (see strategy/indicators.py's realized_vol —
    # a per-bar log-return stdev, not annualized) this sizing scales
    # toward: quantity is multiplied by target_volatility / realized_vol,
    # so a symbol currently calmer than this target gets sized UP (more
    # size for the same dollar-risk budget) and a symbol currently more
    # volatile than this gets sized DOWN. Default is deliberately modest
    # (2%/bar) — tune per bar_interval_minutes and the symbols actually
    # traded; there's no universal right value.
    target_volatility: float = 0.02
    # The inverse-volatility scalar is clamped to this band before being
    # applied, both as a sanity bound (a near-zero realized_vol reading
    # from a very quiet market shouldn't blow the order up to 100x) and
    # because these are multipliers of order_quantity[symbol], which is
    # itself already sized to be reasonable for that symbol/account.
    min_size_multiplier: float = 0.25
    max_size_multiplier: float = 2.0
    # Conviction scaling: a signal that just barely cleared
    # signal_threshold gets min_conviction_multiplier applied; a signal at
    # the maximum possible magnitude (1.0) gets the full 1.0x (before the
    # volatility scalar is layered on top). Linear in between. Set to 1.0
    # to disable conviction scaling while keeping volatility scaling.
    min_conviction_multiplier: float = 0.5


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
    sizing: SizingConfig = field(default_factory=SizingConfig)


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
    # Institutional audit Phase 1.3: maker/limit order execution. See
    # strategy/policy.py (which builds the post-only price) and
    # strategy/engine.py's _manage_resting_order (which runs the
    # place -> wait -> cancel/reprice -> fallback state machine).
    #
    # When True, DecisionPolicy builds LIMIT/post-only orders instead of
    # MARKET orders. False keeps the original always-MARKET behavior —
    # useful for isolating whether a regression is in the maker-order path
    # itself vs. something else.
    use_limit_orders: bool = True
    # How long to let a resting limit order sit before canceling and
    # repricing (or giving up). Kept well under the Rust side's dead-man's
    # switch heartbeat_timeout_secs default (30s) so a strategy managing
    # one slow-to-fill order for a while doesn't itself look dead.
    limit_order_timeout_secs: float = 5.0
    # How many times to cancel-and-reprice at the (possibly moved) best
    # bid/ask before giving up on the maker path for this order. 0 means
    # "place once, cancel on timeout, never reprice."
    limit_reprice_attempts: int = 2
    # If still unfilled after exhausting limit_reprice_attempts, submit a
    # MARKET order for whatever quantity never filled. False means the
    # remaining quantity is simply abandoned — the strategy tried to be a
    # maker and, failing that, does nothing rather than pay taker fees
    # anyway, which defeats the point of this phase for a
    # fee-sensitivity-conscious deployment.
    fallback_to_market: bool = True


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
                sizing=SizingConfig(**strategy_raw.get("sizing", {})),
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
