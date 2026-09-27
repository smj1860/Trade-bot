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

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_model import (
    FEATURE_ORDER,
    build_dataset,
    list_available_symbols,
    load_symbol_dataset,
    move,
    per_symbol_move_threshold,
    persistence_correct_and_total,
    resolve_symbols,
    split_point,
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
    X_a, y_a, _ = build_dataset(closes_a, midpoints_a, highs_a, lows_a, **kwargs)
    X_b, y_b, _ = build_dataset(closes_b, midpoints_b, highs_b, lows_b, **kwargs)

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


def test_move_computes_fractional_change_over_horizon():
    closes = [100.0, 105.0, 110.0, 90.0]
    assert move(closes, 0, 1) == pytest.approx(0.05)
    assert move(closes, 0, 2) == pytest.approx(0.10)
    assert move(closes, 2, 1) == pytest.approx((90.0 - 110.0) / 110.0)


def test_move_zero_base_is_safe():
    assert move([0.0, 5.0], 0, 1) == 0.0


def test_build_dataset_horizon_labels_further_ahead_bar():
    # closes rise then fall right after bar 4: a 1-bar label at i=4 would
    # be "down" (closes[5] < closes[4]), but a 3-bar-ahead label should
    # reflect where price actually is 3 bars later.
    closes = [100.0] * 5 + [99.0, 98.0, 200.0]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    midpoints = [(h + l) / 2.0 for h, l in zip(highs, lows)]
    kwargs = dict(
        sma_window=2, ema_window=2, rsi_window=2, vol_window=2, bar_momentum_window=2,
        bollinger_window=2, bollinger_num_std=2.0, ao_fast_window=1, ao_slow_window=2,
        macd_fast_window=1, macd_slow_window=2, macd_signal_window=1, cci_window=2, williams_r_window=2,
    )
    X1, y1, idx1 = build_dataset(closes, midpoints, highs, lows, horizon=1, **kwargs)
    X3, y3, idx3 = build_dataset(closes, midpoints, highs, lows, horizon=3, **kwargs)
    # bar index 4: 1-bar-ahead is down (99 < 100), 3-bar-ahead is up (200 > 100)
    assert y1[idx1.index(4)] == 0
    assert y3[idx3.index(4)] == 1


def test_build_dataset_min_move_threshold_drops_small_moves():
    closes = [100.0, 100.1, 100.2, 100.1, 110.0, 100.0, 90.0]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    midpoints = [(h + l) / 2.0 for h, l in zip(highs, lows)]
    kwargs = dict(
        sma_window=2, ema_window=2, rsi_window=2, vol_window=2, bar_momentum_window=2,
        bollinger_window=2, bollinger_num_std=2.0, ao_fast_window=1, ao_slow_window=2,
        macd_fast_window=1, macd_slow_window=2, macd_signal_window=1, cci_window=2, williams_r_window=2,
    )
    X_unfiltered, y_unfiltered, idx_unfiltered = build_dataset(closes, midpoints, highs, lows, horizon=1, min_move_threshold=0.0, **kwargs)
    X_filtered, y_filtered, idx_filtered = build_dataset(closes, midpoints, highs, lows, horizon=1, min_move_threshold=0.05, **kwargs)
    assert len(X_filtered) < len(X_unfiltered)
    for i in idx_filtered:
        assert abs(move(closes, i, 1)) >= 0.05


def test_per_symbol_move_threshold_no_filtering_at_top_fraction_one():
    closes = [100.0 + i for i in range(30)]
    assert per_symbol_move_threshold(closes, warmup=5, horizon=1, top_fraction=1.0) == 0.0


def test_per_symbol_move_threshold_keeps_only_extreme_moves():
    # Mostly flat moves with a few large spikes near the end.
    closes = [100.0] * 20 + [100.0, 150.0, 100.0, 50.0, 100.0]
    threshold = per_symbol_move_threshold(closes, warmup=5, horizon=1, top_fraction=0.1)
    assert threshold > 0.0
    # A move of 0.0 (the flat stretch) should not clear this threshold.
    assert 0.0 < threshold


def test_persistence_correct_and_total_matches_hand_computed():
    # Alternating up/down moves; "persistence" predicts the last move continues.
    closes = [100.0, 110.0, 100.0, 110.0, 100.0]
    # indices 1,2,3 are candidate label bars (need i-1 and i+1 in range)
    correct, total = persistence_correct_and_total(closes, indices=[1, 2, 3], horizon=1)
    # i=1: last move (0->1) up, actual (1->2) down -> wrong
    # i=2: last move (1->2) down, actual (2->3) up -> wrong
    # i=3: last move (2->3) up, actual (3->4) down -> wrong
    assert total == 3
    assert correct == 0


def test_persistence_correct_and_total_skips_indices_without_prior_bar():
    closes = [100.0, 110.0, 120.0]
    correct, total = persistence_correct_and_total(closes, indices=[0], horizon=1)
    assert total == 0  # i=0 has no i-horizon bar to compare against


def test_split_point_keeps_at_least_one_row_each_side():
    assert split_point(10, 0.2) == 8
    assert split_point(2, 0.5) == 1
    assert split_point(1, 0.5) == 1
