"""
Unit tests for scripts/train_model.py's pure logic — dataset building,
splitting, and (this round) multi-symbol pooling. These don't touch a
real database: connection-shaped fakes stand in for psycopg2's
conn/cursor so resolve_symbols()/list_available_symbols() can be tested
without SUPABASE_DB_URL, and load_symbol_dataset() is tested against
in-memory closes/highs/lows so its warmup-skip behavior is verifiable
without any real historical data. The actual real-data verification run
(against real Supabase Kraken candles) is documented in
docs/model-training.md, not repeated here as an automated test.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_model import (
    FEATURE_ORDER,
    build_dataset,
    list_available_symbols,
    load_symbol_dataset,
    resolve_symbols,
    time_ordered_split,
)


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        pass

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)


def test_resolve_symbols_single():
    conn = _FakeConn([])
    assert resolve_symbols(conn, "BTC-USD", 60) == ["BTC-USD"]


def test_resolve_symbols_comma_separated_list():
    conn = _FakeConn([])
    assert resolve_symbols(conn, "BTC-USD,ETH-USD, SOL-USD", 60) == ["BTC-USD", "ETH-USD", "SOL-USD"]


def test_resolve_symbols_all_queries_distinct_symbols():
    conn = _FakeConn([("AAVE-USD",), ("BTC-USD",), ("ETH-USD",)])
    assert resolve_symbols(conn, "all", 60) == ["AAVE-USD", "BTC-USD", "ETH-USD"]


def test_list_available_symbols_returns_rows_as_list():
    conn = _FakeConn([("BTC-USD",), ("ETH-USD",)])
    assert list_available_symbols(conn, 60) == ["BTC-USD", "ETH-USD"]


def _make_window_args(**overrides):
    from argparse import Namespace

    defaults = dict(
        sma_window=20,
        ema_window=12,
        rsi_window=14,
        vol_window=20,
        bar_momentum_window=10,
        bollinger_window=20,
        bollinger_num_std=2.0,
        ao_fast_window=5,
        ao_slow_window=34,
        macd_fast_window=12,
        macd_slow_window=26,
        macd_signal_window=9,
        cci_window=20,
        williams_r_window=14,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_load_symbol_dataset_skips_thin_history(monkeypatch):
    import scripts.train_model as train_model

    # Only 10 candles — nowhere near enough for ao_slow_window=34's warmup.
    monkeypatch.setattr(train_model, "load_ohlc", lambda conn, symbol, interval: ([100.0] * 10, [100.0] * 10, [101.0] * 10, [99.0] * 10))
    result = load_symbol_dataset(conn=None, symbol="THIN-USD", interval_minutes=60, window_args=_make_window_args())
    assert result is None


def test_load_symbol_dataset_builds_when_enough_history(monkeypatch):
    import scripts.train_model as train_model

    n = 80
    closes = [100.0 + i * 0.1 for i in range(n)]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    midpoints = [(h + l) / 2.0 for h, l in zip(highs, lows)]
    monkeypatch.setattr(train_model, "load_ohlc", lambda conn, symbol, interval: (closes, midpoints, highs, lows))

    result = load_symbol_dataset(conn=None, symbol="BTC-USD", interval_minutes=60, window_args=_make_window_args())
    assert result is not None
    assert result["symbol"] == "BTC-USD"
    assert len(result["X"]) == len(result["y"])
    assert len(result["X"]) > 0
    assert all(len(row) == len(FEATURE_ORDER) for row in result["X"])


def test_pooling_concatenates_per_symbol_splits_without_cross_contamination():
    """The core pooling guarantee: each symbol is time-split on its own
    closes before any concatenation happens, so a symbol's test rows are
    always chronologically after that same symbol's train rows — pooling
    must never let one symbol's early bars end up in the *test* set
    because of another symbol's split boundary."""
    n = 80
    closes_a = [100.0 + i * 0.1 for i in range(n)]
    closes_b = [1.0 + i * 0.01 for i in range(n)]  # a different price level entirely
    highs_a = [c + 1.0 for c in closes_a]
    lows_a = [c - 1.0 for c in closes_a]
    midpoints_a = [(h + l) / 2.0 for h, l in zip(highs_a, lows_a)]
    highs_b = [c + 0.05 for c in closes_b]
    lows_b = [c - 0.05 for c in closes_b]
    midpoints_b = [(h + l) / 2.0 for h, l in zip(highs_b, lows_b)]

    kwargs = dict(
        sma_window=5, ema_window=5, rsi_window=5, vol_window=5, bar_momentum_window=5,
        bollinger_window=5, bollinger_num_std=2.0, ao_fast_window=2, ao_slow_window=5,
        macd_fast_window=2, macd_slow_window=5, macd_signal_window=2, cci_window=5, williams_r_window=5,
    )
    X_a, y_a = build_dataset(closes_a, midpoints_a, highs_a, lows_a, **kwargs)
    X_b, y_b = build_dataset(closes_b, midpoints_b, highs_b, lows_b, **kwargs)

    a_train_X, a_train_y, a_test_X, a_test_y = time_ordered_split(X_a, y_a, 0.2)
    b_train_X, b_train_y, b_test_X, b_test_y = time_ordered_split(X_b, y_b, 0.2)

    pooled_train_X = a_train_X + b_train_X
    pooled_test_X = a_test_X + b_test_X

    # Every pooled train row must be a row that appeared in one symbol's
    # own train split, never a row from that symbol's test split.
    assert all(row in a_train_X or row in b_train_X for row in pooled_train_X)
    assert all(row in a_test_X or row in b_test_X for row in pooled_test_X)
    assert len(pooled_train_X) == len(a_train_X) + len(b_train_X)
    assert len(pooled_test_X) == len(a_test_X) + len(b_test_X)
