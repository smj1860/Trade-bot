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
    _embargo_test_start,
    _purge_train_end,
    build_dataset,
    holdout_boundary,
    list_available_symbols,
    load_symbol_dataset,
    move,
    net_pnl,
    per_symbol_move_threshold,
    persistence_correct_and_total,
    persistence_correct_and_total_from_labels,
    resolve_symbols,
    seal_holdout,
    simulate_net_pnl,
    simulate_triple_barrier_net_pnl,
    split_point,
    time_ordered_split,
    triple_barrier_label,
    triple_barrier_net_pnl,
    triple_barrier_touch,
    walk_forward_splits,
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
    monkeypatch.setattr(train_model, "load_ohlc", lambda conn, symbol, interval: ([100.0] * 10, [100.0] * 10, [101.0] * 10, [99.0] * 10, [1.0] * 10))
    result = load_symbol_dataset(conn=None, symbol="THIN-USD", interval_minutes=60, window_args=_make_window_args())
    assert result is None


def test_load_symbol_dataset_builds_when_enough_history(monkeypatch):
    import scripts.train_model as train_model

    n = 80
    closes = [100.0 + i * 0.1 for i in range(n)]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    midpoints = [(h + l) / 2.0 for h, l in zip(highs, lows)]
    volumes = [1.0] * len(closes)
    monkeypatch.setattr(train_model, "load_ohlc", lambda conn, symbol, interval: (closes, midpoints, highs, lows, volumes))

    # min_move=0.0 explicitly: this test is about the warmup-skip behavior,
    # not about the fee-derived default threshold (net-P&L labeling is
    # covered separately below) — the synthetic closes here move ~0.1%/bar,
    # far below the default derived threshold, which would filter every row.
    result = load_symbol_dataset(conn=None, symbol="BTC-USD", interval_minutes=60, window_args=_make_window_args(min_move=0.0))
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
    volumes_a = [1.0] * len(closes_a)
    volumes_b = [1.0] * len(closes_b)
    X_a, y_a, _ = build_dataset(closes_a, midpoints_a, highs_a, lows_a, volumes_a, **kwargs)
    X_b, y_b, _ = build_dataset(closes_b, midpoints_b, highs_b, lows_b, volumes_b, **kwargs)

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
    volumes = [1.0] * len(closes)
    X1, y1, idx1 = build_dataset(closes, midpoints, highs, lows, volumes, horizon=1, **kwargs)
    X3, y3, idx3 = build_dataset(closes, midpoints, highs, lows, volumes, horizon=3, **kwargs)
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
    volumes = [1.0] * len(closes)
    X_unfiltered, y_unfiltered, idx_unfiltered = build_dataset(closes, midpoints, highs, lows, volumes, horizon=1, min_move_threshold=0.0, **kwargs)
    X_filtered, y_filtered, idx_filtered = build_dataset(closes, midpoints, highs, lows, volumes, horizon=1, min_move_threshold=0.05, **kwargs)
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


def test_net_pnl_long_call_subtracts_round_trip_cost():
    closes = [100.0, 105.0]  # +5% move
    # Called "up" (long) and it went up: raw return is the move itself,
    # minus the round-trip cost.
    assert net_pnl(closes, 0, 1, predicted_up=True, round_trip_cost=0.016) == pytest.approx(0.05 - 0.016)


def test_net_pnl_short_call_inverts_the_move():
    closes = [100.0, 105.0]  # +5% move
    # Called "down" (short) but it went up: the trade loses the move's
    # magnitude, then still pays the round-trip cost.
    assert net_pnl(closes, 0, 1, predicted_up=False, round_trip_cost=0.016) == pytest.approx(-0.05 - 0.016)


def test_net_pnl_short_call_correct_direction_is_profitable_after_cost():
    closes = [100.0, 90.0]  # -10% move
    assert net_pnl(closes, 0, 1, predicted_up=False, round_trip_cost=0.016) == pytest.approx(0.10 - 0.016)


def test_simulate_net_pnl_aggregates_total_and_count_across_rows():
    closes = [100.0, 105.0, 110.0, 90.0]
    # i=0: move=+5%, predicted up (1) -> 0.05 - cost
    # i=1: move=+4.76...%, predicted down (0) -> -(that move) - cost
    # i=2: move=(90-110)/110=-18.18...%, predicted down (0) -> +0.1818... - cost
    indices = [0, 1, 2]
    predictions = [1, 0, 0]
    cost = 0.01
    total, count = simulate_net_pnl(closes, indices, 1, predictions, cost)
    expected_total = (
        net_pnl(closes, 0, 1, predicted_up=True, round_trip_cost=cost)
        + net_pnl(closes, 1, 1, predicted_up=False, round_trip_cost=cost)
        + net_pnl(closes, 2, 1, predicted_up=False, round_trip_cost=cost)
    )
    assert count == 3
    assert total == pytest.approx(expected_total)


def test_simulate_net_pnl_empty_indices_returns_zero_count():
    total, count = simulate_net_pnl([100.0, 101.0], [], 1, [], 0.01)
    assert total == 0.0
    assert count == 0


def test_load_symbol_dataset_derives_min_move_from_fees_by_default(monkeypatch):
    """With --min-move omitted (None, the default), the effective threshold
    should be 2x taker_fee + profit_margin, not 0.0 — this is the "simulated
    paper trading" behavior: a move that doesn't clear real round-trip costs
    isn't a trade worth labeling, even if a smaller move would have been
    a fine label under the old flat-threshold-free default."""
    import scripts.train_model as train_model

    n = 80
    # Alternate small (~0.1%) and large (~5%) moves so filtering actually
    # changes which rows survive.
    closes = []
    price = 100.0
    for i in range(n):
        price *= 1.05 if i % 5 == 0 else 1.001
        closes.append(price)
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    midpoints = [(h + l) / 2.0 for h, l in zip(highs, lows)]
    volumes = [1.0] * len(closes)
    monkeypatch.setattr(train_model, "load_ohlc", lambda conn, symbol, interval: (closes, midpoints, highs, lows, volumes))

    unfiltered = load_symbol_dataset(conn=None, symbol="BTC-USD", interval_minutes=60, window_args=_make_window_args(min_move=0.0))
    derived = load_symbol_dataset(conn=None, symbol="BTC-USD", interval_minutes=60, window_args=_make_window_args(taker_fee=0.008, slippage=0.0005, profit_margin=0.0))

    assert derived is not None and unfiltered is not None
    # 2 * (taker_fee + slippage) = 2 * (0.008 + 0.0005) = 0.017
    assert derived["min_move_threshold"] == pytest.approx(0.017)
    assert derived["round_trip_cost"] == pytest.approx(0.017)
    assert len(derived["X"]) < len(unfiltered["X"])
    for i in derived["indices"]:
        assert abs(move(closes, i, 1)) >= 0.017


def test_load_symbol_dataset_explicit_min_move_overrides_fee_derivation(monkeypatch):
    import scripts.train_model as train_model

    n = 80
    closes = [100.0 + i * 0.1 for i in range(n)]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    midpoints = [(h + l) / 2.0 for h, l in zip(highs, lows)]
    volumes = [1.0] * len(closes)
    monkeypatch.setattr(train_model, "load_ohlc", lambda conn, symbol, interval: (closes, midpoints, highs, lows, volumes))

    result = load_symbol_dataset(conn=None, symbol="BTC-USD", interval_minutes=60, window_args=_make_window_args(min_move=0.0, taker_fee=0.008))
    assert result is not None
    assert result["min_move_threshold"] == pytest.approx(0.0)


def test_walk_forward_splits_expanding_window_train_grows_each_fold():
    # 10 rows, 4 folds -> 5 blocks of 2 rows each: block 1 is the initial
    # train, blocks 2-5 are each fold's test block in turn.
    X = [[float(i)] for i in range(10)]
    y = [i % 2 for i in range(10)]
    indices = list(range(10))

    folds = list(walk_forward_splits(X, y, indices, n_folds=4))
    assert len(folds) == 4

    train_sizes = [len(f[0]) for f in folds]
    test_sizes = [len(f[3]) for f in folds]
    assert train_sizes == [2, 4, 6, 8]  # strictly expanding
    assert test_sizes == [2, 2, 2, 2]


def test_walk_forward_splits_train_never_overlaps_test_chronologically():
    X = [[float(i)] for i in range(12)]
    y = [i % 2 for i in range(12)]
    indices = list(range(12))

    for X_train, y_train, idx_train, X_test, y_test, idx_test in walk_forward_splits(X, y, indices, n_folds=3):
        assert max(idx_train) < min(idx_test)  # no lookahead leakage
        assert len(X_train) == len(y_train) == len(idx_train)
        assert len(X_test) == len(y_test) == len(idx_test)


def test_walk_forward_splits_successive_folds_test_different_periods():
    X = [[float(i)] for i in range(10)]
    y = [i % 2 for i in range(10)]
    indices = list(range(10))

    folds = list(walk_forward_splits(X, y, indices, n_folds=4))
    test_index_sets = [set(f[5]) for f in folds]
    # Every fold's test block is disjoint from every other fold's.
    for a in range(len(test_index_sets)):
        for b in range(a + 1, len(test_index_sets)):
            assert test_index_sets[a].isdisjoint(test_index_sets[b])


def test_walk_forward_splits_too_little_data_yields_nothing():
    X = [[1.0], [2.0], [3.0]]
    y = [0, 1, 0]
    indices = [0, 1, 2]
    # 3 rows, 5 folds -> block_size = 3 // 6 = 0 -> no usable folds.
    assert list(walk_forward_splits(X, y, indices, n_folds=5)) == []


def test_walk_forward_splits_remainder_absorbed_into_final_test_block():
    # 11 rows, 2 folds -> 3 blocks of floor(11/3)=3 rows, with the leftover
    # 2 rows absorbed into the final (last fold's test) block rather than
    # dropped.
    X = [[float(i)] for i in range(11)]
    y = [i % 2 for i in range(11)]
    indices = list(range(11))

    folds = list(walk_forward_splits(X, y, indices, n_folds=2))
    assert len(folds) == 2
    assert len(folds[0][3]) == 3          # first fold's test block: exactly one block
    assert len(folds[1][3]) == 11 - 3 - 3  # second (last) fold's test block absorbs the remainder
    assert sum(len(f[3]) for f in folds) + len(folds[0][0]) == 11  # every row accounted for


def test_purge_train_end_no_embargo_is_a_no_op():
    indices = list(range(10))
    assert _purge_train_end(indices, train_end=6, test_start_bar=6, embargo=0) == 6


def test_purge_train_end_drops_rows_reaching_the_boundary():
    indices = list(range(10))
    # Rows 4 and 5 both have index + 2 >= 6 (test_start_bar); row 3 (3+2=5 < 6) survives.
    assert _purge_train_end(indices, train_end=6, test_start_bar=6, embargo=2) == 4


def test_purge_train_end_can_empty_the_train_set():
    indices = list(range(10))
    assert _purge_train_end(indices, train_end=6, test_start_bar=6, embargo=100) == 0


def test_embargo_test_start_no_embargo_is_a_no_op():
    indices = list(range(10))
    assert _embargo_test_start(indices, test_start=6, test_end=10, test_start_bar=6, embargo=0) == 6


def test_embargo_test_start_drops_leading_rows_near_the_boundary():
    indices = list(range(10))
    # Rows 6 and 7 are within 2 of the boundary (6); row 8 (8 >= 6+2) survives.
    assert _embargo_test_start(indices, test_start=6, test_end=10, test_start_bar=6, embargo=2) == 8


def test_embargo_test_start_can_empty_the_test_set():
    indices = list(range(10))
    assert _embargo_test_start(indices, test_start=6, test_end=10, test_start_bar=6, embargo=100) == 10


def test_walk_forward_splits_embargo_zero_matches_no_embargo_behavior():
    # Default embargo=0 must reproduce the exact pre-embargo behavior —
    # backward compatibility for every existing call site/test above.
    X = [[float(i)] for i in range(10)]
    y = [i % 2 for i in range(10)]
    indices = list(range(10))
    assert list(walk_forward_splits(X, y, indices, n_folds=4)) == list(
        walk_forward_splits(X, y, indices, n_folds=4, embargo=0)
    )


def test_walk_forward_splits_embargo_purges_trailing_train_rows_near_boundary():
    # A training row's label can look up to `embargo` bars ahead — any
    # training row whose index + embargo reaches at or past the first test
    # bar must be purged, since its label could have been computed from
    # data inside the test block. Use a later fold (more accumulated
    # train rows) so purging a couple of trailing ones still leaves a
    # non-empty train set.
    X = [[float(i)] for i in range(30)]
    y = [i % 2 for i in range(30)]
    indices = list(range(30))

    no_embargo_fold = list(walk_forward_splits(X, y, indices, n_folds=4))[-1]
    with_embargo_fold = list(walk_forward_splits(X, y, indices, n_folds=4, embargo=2))[-1]
    assert len(with_embargo_fold[2]) < len(no_embargo_fold[2])  # idx_train shrank
    test_start_bar = with_embargo_fold[5][0]
    for i in with_embargo_fold[2]:
        assert i + 2 < test_start_bar


def test_walk_forward_splits_embargo_leaves_no_train_label_overlapping_test():
    # The property embargo exists to guarantee, checked directly: for
    # every purged fold, no surviving training row's label window
    # (index..index+embargo) reaches into the test block's first index.
    X = [[float(i)] for i in range(30)]
    y = [i % 2 for i in range(30)]
    indices = list(range(30))
    embargo = 3

    for X_train, y_train, idx_train, X_test, y_test, idx_test in walk_forward_splits(
        X, y, indices, n_folds=4, embargo=embargo
    ):
        if not idx_test:
            continue
        test_start_bar = idx_test[0]
        for i in idx_train:
            assert i + embargo < test_start_bar


def test_walk_forward_splits_embargo_purges_leading_test_rows_near_boundary():
    # The test-side half of purge+embargo: test rows within `embargo` bars
    # of the boundary are dropped too, as a buffer against serial
    # correlation across it (not just literal label overlap).
    X = [[float(i)] for i in range(10)]
    y = [i % 2 for i in range(10)]
    indices = list(range(10))

    no_embargo = list(walk_forward_splits(X, y, indices, n_folds=4))[0]
    with_embargo = list(walk_forward_splits(X, y, indices, n_folds=4, embargo=1))[0]
    assert len(with_embargo[3]) <= len(no_embargo[3])  # X_test shrank or stayed the same
    assert min(with_embargo[5]) >= min(no_embargo[5]) + 1


def test_walk_forward_splits_large_embargo_can_starve_a_fold():
    # An embargo that consumes an entire block should just drop that fold
    # (empty train or test) rather than yield a degenerate split.
    X = [[float(i)] for i in range(10)]
    y = [i % 2 for i in range(10)]
    indices = list(range(10))
    # Block size is 2; an embargo of 100 purges every training row in
    # every fold's would-be train set.
    assert list(walk_forward_splits(X, y, indices, n_folds=4, embargo=100)) == []


def test_walk_forward_splits_embargo_respects_gaps_in_filtered_indices():
    # indices need not be contiguous (min-move/triple-barrier filtering
    # leaves gaps) — embargo must compare real bar-index distance, not
    # row-count distance, so a gap right at the boundary is handled
    # correctly rather than under- or over-purging.
    X = [[float(i)] for i in range(8)]
    y = [i % 2 for i in range(8)]
    # A gap: bar 10 immediately follows bar 3 (bars 4-9 were filtered out
    # upstream) — row-count spacing would suggest embargo=2 purges only
    # the last row, but the real bar-index gap is much larger.
    indices = [0, 1, 2, 3, 10, 11, 12, 13]

    folds = list(walk_forward_splits(X, y, indices, n_folds=3, embargo=2))
    for X_train, y_train, idx_train, X_test, y_test, idx_test in folds:
        if not idx_test or not idx_train:
            continue
        assert max(idx_train) + 2 < idx_test[0]


def test_triple_barrier_touch_upper_touched_via_high_not_close():
    # Bar 2 spikes above the upper barrier intrabar (high=106) even though
    # its close (102) never gets there — the touch must be detected from
    # the real high, not the close.
    closes = [100.0, 101.0, 102.0, 999.0, 999.0]
    highs = [100.0, 101.0, 106.0, 999.0, 999.0]
    lows = [100.0, 99.0, 100.0, 999.0, 999.0]
    touch, ret = triple_barrier_touch(highs, lows, closes, i=0, max_hold=4, barrier_pct=0.05)
    assert touch == "upper"
    assert ret == pytest.approx(0.05)


def test_triple_barrier_touch_lower_touched_via_low_not_close():
    closes = [100.0, 101.0, 96.0, 999.0, 999.0]
    highs = [100.0, 101.0, 102.0, 999.0, 999.0]
    lows = [100.0, 99.0, 94.0, 999.0, 999.0]
    touch, ret = triple_barrier_touch(highs, lows, closes, i=0, max_hold=4, barrier_pct=0.05)
    assert touch == "lower"
    assert ret == pytest.approx(-0.05)


def test_triple_barrier_touch_earlier_bar_wins_even_if_later_bar_touches_other_side():
    # Bar 1 already touches the lower barrier; bar 2 touching the upper
    # barrier afterward must not matter — the first touch along the path wins.
    closes = [100.0, 94.0, 106.0, 999.0]
    highs = [100.0, 95.0, 106.0, 999.0]
    lows = [100.0, 94.0, 105.0, 999.0]
    touch, ret = triple_barrier_touch(highs, lows, closes, i=0, max_hold=3, barrier_pct=0.05)
    assert touch == "lower"
    assert ret == pytest.approx(-0.05)


def test_triple_barrier_touch_ambiguous_same_bar_returns_none():
    # A single bar's high and low both cross their respective barriers —
    # real OHLC data can't say which the price reached first intrabar.
    closes = [100.0, 100.0]
    highs = [100.0, 106.0]
    lows = [100.0, 94.0]
    touch, ret = triple_barrier_touch(highs, lows, closes, i=0, max_hold=1, barrier_pct=0.05)
    assert touch is None
    assert ret is None


def test_triple_barrier_touch_timeout_marks_at_final_close():
    # Neither barrier is touched within max_hold bars — mark at the
    # vertical barrier's close instead.
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    highs = [100.0, 101.0, 102.0, 103.0, 104.0]
    lows = [100.0, 100.0, 101.0, 102.0, 103.0]
    touch, ret = triple_barrier_touch(highs, lows, closes, i=0, max_hold=4, barrier_pct=0.05)
    assert touch == "timeout"
    assert ret == pytest.approx(0.04)


def test_triple_barrier_touch_insufficient_history_returns_none():
    closes = [100.0, 101.0, 102.0]
    highs = closes[:]
    lows = closes[:]
    touch, ret = triple_barrier_touch(highs, lows, closes, i=0, max_hold=5, barrier_pct=0.05)
    assert touch is None
    assert ret is None


def test_triple_barrier_touch_zero_entry_price_returns_none():
    closes = [0.0, 1.0, 2.0]
    highs = closes[:]
    lows = closes[:]
    touch, ret = triple_barrier_touch(highs, lows, closes, i=0, max_hold=2, barrier_pct=0.05)
    assert touch is None
    assert ret is None


def test_triple_barrier_label_matches_touch_side():
    closes = [100.0, 101.0, 102.0, 999.0]
    highs_up = [100.0, 101.0, 106.0, 999.0]
    lows_up = [100.0, 99.0, 100.0, 999.0]
    assert triple_barrier_label(highs_up, lows_up, closes, i=0, max_hold=3, barrier_pct=0.05) == 1

    highs_down = [100.0, 101.0, 102.0, 999.0]
    lows_down = [100.0, 99.0, 94.0, 999.0]
    assert triple_barrier_label(highs_down, lows_down, closes, i=0, max_hold=3, barrier_pct=0.05) == 0


def test_triple_barrier_label_none_on_timeout_or_ambiguous():
    closes = [100.0, 101.0, 102.0, 103.0]
    flat_highs = [100.0, 101.0, 102.0, 103.0]
    flat_lows = [100.0, 100.0, 101.0, 102.0]
    assert triple_barrier_label(flat_highs, flat_lows, closes, i=0, max_hold=3, barrier_pct=0.05) is None

    ambiguous_highs = [100.0, 106.0]
    ambiguous_lows = [100.0, 94.0]
    assert triple_barrier_label(ambiguous_highs, ambiguous_lows, [100.0, 100.0], i=0, max_hold=1, barrier_pct=0.05) is None


def test_triple_barrier_net_pnl_long_call_on_upper_touch_subtracts_cost():
    closes = [100.0, 101.0, 106.0, 999.0]
    highs = [100.0, 101.0, 106.0, 999.0]
    lows = [100.0, 99.0, 100.0, 999.0]
    pnl = triple_barrier_net_pnl(highs, lows, closes, i=0, max_hold=3, barrier_pct=0.05, predicted_up=True, round_trip_cost=0.017)
    assert pnl == pytest.approx(0.05 - 0.017)


def test_triple_barrier_net_pnl_short_call_on_upper_touch_loses_the_move():
    # Called "down" (short) but price touched the upper barrier first: the
    # trade loses the barrier's magnitude, then still pays round-trip cost.
    closes = [100.0, 101.0, 106.0, 999.0]
    highs = [100.0, 101.0, 106.0, 999.0]
    lows = [100.0, 99.0, 100.0, 999.0]
    pnl = triple_barrier_net_pnl(highs, lows, closes, i=0, max_hold=3, barrier_pct=0.05, predicted_up=False, round_trip_cost=0.017)
    assert pnl == pytest.approx(-0.05 - 0.017)


def test_triple_barrier_net_pnl_none_when_touch_unresolved():
    closes = [100.0, 101.0, 102.0]
    highs = closes[:]
    lows = closes[:]
    assert triple_barrier_net_pnl(highs, lows, closes, i=0, max_hold=5, barrier_pct=0.05, predicted_up=True, round_trip_cost=0.017) is None


def test_simulate_triple_barrier_net_pnl_aggregates_and_skips_unresolved():
    # 7 bars so both index 0 and index 3 have a full max_hold=3 window to
    # walk forward over.
    closes = [100.0, 101.0, 106.0, 101.0, 96.0, 999.0, 999.0]
    highs = [100.0, 101.0, 106.0, 101.0, 102.0, 999.0, 999.0]
    lows = [100.0, 99.0, 100.0, 99.0, 95.0, 999.0, 999.0]
    # index 0: entry 100, upper barrier 105 touched at bar 2 (high=106) ->
    #   predicted up (correct) -> +0.05 - cost
    # index 3: entry 101, lower barrier 95.95 touched at bar 4 (low=95) ->
    #   predicted up (wrong) -> -0.05 - cost
    total, count = simulate_triple_barrier_net_pnl(
        highs, lows, closes, indices=[0, 3], horizon=3, barrier_pct=0.05, predictions=[1, 1], round_trip_cost=0.017
    )
    assert count == 2
    assert total == pytest.approx((0.05 - 0.017) + (-0.05 - 0.017))


def test_persistence_correct_and_total_from_labels_scores_against_given_labels():
    closes = [100.0, 90.0, 110.0, 95.0, 120.0]
    # horizon=1: predicted_up at i uses closes[i] vs closes[i-1].
    # i=2: closes[2]=110 > closes[1]=90 -> predicted up (True)
    # i=3: closes[3]=95 < closes[2]=110 -> predicted up (False)
    indices = [2, 3]
    y = [1, 0]  # matches predictions exactly -> both correct
    correct, total = persistence_correct_and_total_from_labels(closes, indices, y, horizon=1)
    assert (correct, total) == (2, 2)

    y_wrong = [0, 1]  # both wrong relative to persistence's prediction
    correct, total = persistence_correct_and_total_from_labels(closes, indices, y_wrong, horizon=1)
    assert (correct, total) == (0, 2)


def test_persistence_correct_and_total_from_labels_skips_out_of_range_index():
    closes = [100.0, 101.0]
    correct, total = persistence_correct_and_total_from_labels(closes, indices=[0], y=[1], horizon=1)
    assert (correct, total) == (0, 0)  # i - horizon < 0, skipped


def test_build_dataset_triple_barrier_scheme_drops_timeouts_and_labels_by_touch(monkeypatch):
    # A long enough synthetic series with a clear upper-touch bar and a
    # clear "nothing much happens" (timeout) stretch, to confirm
    # build_dataset's triple-barrier branch labels the former and drops the
    # latter, using highs/lows rather than move()'s close-only endpoint.
    import scripts.train_model as train_model

    n = 60
    closes = [100.0 + (i % 3) * 0.01 for i in range(n)]  # nearly flat -> mostly timeouts
    highs = list(closes)
    lows = list(closes)
    # Engineer one clear upper-barrier touch a few bars after warmup ends.
    touch_i = 40
    highs[touch_i + 2] = closes[touch_i] * 1.10  # well past a 5% barrier
    lows[touch_i + 2] = closes[touch_i] * 1.09

    warmup = train_model.dataset_warmup(
        sma_window=5, ema_window=5, rsi_window=5, vol_window=5, bar_momentum_window=5,
        bollinger_window=5, ao_slow_window=5, macd_slow_window=5, macd_signal_window=3,
        cci_window=5, williams_r_window=5,
    )
    midpoints = [(h + l) / 2.0 for h, l in zip(highs, lows)]
    volumes = [1.0] * n

    X, y, indices = build_dataset(
        closes, midpoints, highs, lows, volumes,
        sma_window=5, ema_window=5, rsi_window=5, vol_window=5, bar_momentum_window=5,
        bollinger_window=5, bollinger_num_std=2.0, ao_fast_window=3, ao_slow_window=5,
        macd_fast_window=3, macd_slow_window=5, macd_signal_window=3, cci_window=5, williams_r_window=5,
        horizon=4, min_move_threshold=0.05, label_scheme="triple-barrier",
    )
    assert touch_i in indices  # the engineered upper-touch bar survived filtering
    assert y[indices.index(touch_i)] == 1
    # Most of the flat stretch should have timed out and been dropped —
    # far fewer surviving rows than warmup-to-end candidate bars.
    assert len(X) < (n - warmup - 4)


# --- Institutional audit Phase 2.5: --holdout-days sealing ---


def test_holdout_boundary_disabled_returns_full_length():
    assert holdout_boundary(n_bars=1000, holdout_days=0, interval_minutes=60) == 1000


def test_holdout_boundary_seals_the_expected_number_of_bars():
    # 60-minute candles, 10 days = 240 bars.
    assert holdout_boundary(n_bars=1000, holdout_days=10, interval_minutes=60) == 760


def test_holdout_boundary_respects_interval():
    # 15-minute candles, 1 day = 96 bars.
    assert holdout_boundary(n_bars=200, holdout_days=1, interval_minutes=15) == 104


def test_holdout_boundary_clamps_to_zero_when_holdout_exceeds_history():
    assert holdout_boundary(n_bars=50, holdout_days=1000, interval_minutes=60) == 0


def test_seal_holdout_truncates_all_five_series_identically():
    closes = list(range(100))
    midpoints = [c + 0.5 for c in closes]
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]
    volumes = [1.0] * 100

    sealed = seal_holdout(closes, midpoints, highs, lows, volumes, holdout_days=1, interval_minutes=24)
    # 1 day at 24-minute bars = 60 bars sealed off -> 40 remain.
    for series in sealed:
        assert len(series) == 40
    sealed_closes, sealed_mids, sealed_highs, sealed_lows, sealed_vols = sealed
    assert sealed_closes == closes[:40]
    assert sealed_mids == midpoints[:40]
    assert sealed_highs == highs[:40]
    assert sealed_lows == lows[:40]
    assert sealed_vols == volumes[:40]


def test_seal_holdout_is_a_no_op_when_holdout_days_is_zero():
    closes = list(range(50))
    midpoints, highs, lows, volumes = closes[:], closes[:], closes[:], [1.0] * 50
    sealed = seal_holdout(closes, midpoints, highs, lows, volumes, holdout_days=0, interval_minutes=60)
    assert sealed == (closes, midpoints, highs, lows, volumes)


def test_load_symbol_dataset_never_sees_the_sealed_holdout_window(monkeypatch):
    import scripts.train_model as train_model

    n = 200
    closes = [100.0 + i * 0.1 for i in range(n)]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    midpoints = [(h + l) / 2.0 for h, l in zip(highs, lows)]
    volumes = [1.0] * n
    monkeypatch.setattr(train_model, "load_ohlc", lambda conn, symbol, interval: (closes, midpoints, highs, lows, volumes))

    # 60-minute bars, 10 days = 240 bars sealed -- more than the whole
    # 200-bar series, so everything should be sealed off (boundary clamps
    # to 0) and the dataset should come back None (too little history).
    sealed_all = load_symbol_dataset(
        conn=None, symbol="BTC-USD", interval_minutes=60,
        window_args=_make_window_args(min_move=0.0, holdout_days=10),
    )
    assert sealed_all is None

    # A smaller holdout (1 day = 24 bars) should leave a visibly shorter
    # `closes` series than the unsealed run -- proof the tail is actually
    # gone, not merely unused by feature/label computation.
    unsealed = load_symbol_dataset(
        conn=None, symbol="BTC-USD", interval_minutes=60,
        window_args=_make_window_args(min_move=0.0, holdout_days=0),
    )
    sealed_small = load_symbol_dataset(
        conn=None, symbol="BTC-USD", interval_minutes=60,
        window_args=_make_window_args(min_move=0.0, holdout_days=1),
    )
    assert unsealed is not None and sealed_small is not None
    assert len(sealed_small["closes"]) == len(unsealed["closes"]) - 24
    assert sealed_small["closes"] == closes[:-24]


def test_parse_intervals_single_and_list():
    from scripts.train_model import parse_intervals

    assert parse_intervals("60") == [60]
    assert parse_intervals(60) == [60]
    assert parse_intervals("60, 240,360,1440") == [60, 240, 360, 1440]
    assert parse_intervals("60,60,240") == [60, 240]


def test_parse_intervals_rejects_bad_values():
    from scripts.train_model import parse_intervals

    with pytest.raises(ValueError):
        parse_intervals("")
    with pytest.raises(ValueError):
        parse_intervals("0")
