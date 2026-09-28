"""
Unit tests for strategy/config.py's TOML loading — in particular
Institutional audit Phase 2.1's new [strategy.sizing] table, which must
both parse real values from a config file and default sensibly when a
config predates it (an older strategy_config.toml with no [strategy.sizing]
section at all shouldn't break).
"""

import tempfile
from decimal import Decimal
from pathlib import Path

from strategy.config import Config, SizingConfig

_MINIMAL_TOML = """
[connection]
rust_core_addr = "localhost:50051"

[strategy]
name = "test"
symbols = ["BTC-USD"]
exchange = "kraken"
signal_threshold = 0.3
cooldown_seconds = 10

[strategy.order_quantity]
BTC-USD = "0.01"

[strategy.features]
momentum_window = 20

[strategy.model]
kind = "rule_based"

[portfolio]
[portfolio.max_position]
BTC-USD = "0.05"

[logging]
log_path = "logs/strategy.jsonl"
level = "INFO"
"""

_SIZING_TOML = (
    _MINIMAL_TOML
    + """
[strategy.sizing]
enabled = false
target_volatility = 0.03
min_size_multiplier = 0.1
max_size_multiplier = 3.0
min_conviction_multiplier = 0.2
"""
)


def _load(toml_text: str) -> Config:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_text)
        path = f.name
    try:
        return Config.load(path)
    finally:
        Path(path).unlink()


def test_sizing_defaults_when_section_is_absent():
    config = _load(_MINIMAL_TOML)
    assert config.strategy.sizing == SizingConfig()
    assert config.strategy.sizing.enabled is True
    assert config.strategy.sizing.target_volatility == 0.02


def test_sizing_loads_explicit_values():
    config = _load(_SIZING_TOML)
    sizing = config.strategy.sizing
    assert sizing.enabled is False
    assert sizing.target_volatility == 0.03
    assert sizing.min_size_multiplier == 0.1
    assert sizing.max_size_multiplier == 3.0
    assert sizing.min_conviction_multiplier == 0.2


def test_order_quantity_still_parses_as_decimal():
    config = _load(_MINIMAL_TOML)
    assert config.strategy.order_quantity["BTC-USD"] == Decimal("0.01")
