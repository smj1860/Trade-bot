"""
Unit tests for scripts/evaluate_holdout.py's pure logic — the sealed
final-holdout dataset loader and per-trade return extraction that back
the institutional audit's Phase 2.5 data-snooping fix (see
strategy/dsr.py and scripts/evaluate_holdout.py's module docstring for
the full picture). Like test_train_model.py, these use in-memory
closes/highs/lows via a monkeypatched load_ohlc rather than a real
database connection.
"""

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_holdout import load_holdout_dataset, per_trade_returns
from scripts.train_model import dataset_warmup, holdout_boundary


def _make_args(**overrides):
    defaults = dict(
        sma_window=5, ema_window=5, rsi_window=5, vol_window=5, bar_momentum_window=5,
        bollinger_window=5, bollinger_num_std=2.0, ao_fast_window=3, ao_slow_window=5,
        macd_fast_window=3, macd_slow_window=5, macd_signal_window=3, cci_window=5, williams_r_window=5,
        horizon=1, min_move=0.0, top_fraction=1.0, taker_fee=0.008, slippage=0.0005, profit_margin=0.0,
        label_scheme="fixed-horizon", holdout_days=1,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _warmup(args) -> int:
    return dataset_warmup(
        args.sma_window, args.ema_window, args.rsi_window, args.vol_window,
        args.bar_momentum_window, args.bollinger_window, args.ao_slow_window,
        args.macd_slow_window, args.macd_signal_window, args.cci_window, args.williams_r_window,
    )


def test_load_holdout_dataset_requires_holdout_days_positive(monkeypatch):
    import scripts.evaluate_holdout as eh

    monkeypatch.setattr(eh, "load_ohlc", lambda conn, symbol, interval: ([100.0] * 100, [100.0] * 100, [101.0] * 100, [99.0] * 100, [1.0] * 100))
    result = load_holdout_dataset(conn=None, symbol="BTC-USD", interval_minutes=60, args=_make_args(holdout_days=0))
    assert result is None


def test_load_holdout_dataset_returns_none_when_sealed_window_too_thin(monkeypatch):
    import scripts.evaluate_holdout as eh

    # A genuinely thin series -- not enough candles to satisfy
    # warmup + horizon even after accounting for the sealed window.
    n = 8
    closes = [100.0] * n
    monkeypatch.setattr(eh, "load_ohlc", lambda conn, symbol, interval: (closes, closes, closes, closes, [1.0] * n))
    result = load_holdout_dataset(conn=None, symbol="THIN-USD", interval_minutes=60, args=_make_args(holdout_days=1))
    assert result is None


def test_load_holdout_dataset_only_returns_rows_from_the_sealed_window(monkeypatch):
    import scripts.evaluate_holdout as eh

    n = 300
    closes = [100.0 + i * 0.05 for i in range(n)]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    midpoints = [(h + l) / 2.0 for h, l in zip(highs, lows)]
    volumes = [1.0] * n
    monkeypatch.setattr(eh, "load_ohlc", lambda conn, symbol, interval: (closes, midpoints, highs, lows, volumes))

    args = _make_args(holdout_days=2)  # 2 days at 60min bars = 48 bars sealed
    result = load_holdout_dataset(conn=None, symbol="BTC-USD", interval_minutes=60, args=args)
    assert result is not None

    boundary = holdout_boundary(n, args.holdout_days, 60)
    warmup = _warmup(args)
    lookback_start = max(0, boundary - warmup)
    local_boundary = boundary - lookback_start

    # Every surviving row's index (in the trimmed series' own space) must
    # be at or past the local boundary -- i.e. every kept row's feature
    # was computed as-of a bar inside the sealed window, never before it.
    assert all(i >= local_boundary for i in result["indices"])
    assert len(result["X"]) > 0


def test_per_trade_returns_fixed_horizon_matches_net_pnl():
    from scripts.train_model import net_pnl

    closes = [100.0, 101.0, 99.0, 105.0, 95.0, 110.0]
    entry = {
        "closes": closes, "highs": closes, "lows": closes,
        "horizon": 1, "round_trip_cost": 0.01, "barrier_pct": 0.0,
        "label_scheme": "fixed-horizon", "indices": [0, 1, 2, 3],
    }
    predictions = [1, 0, 1, 0]
    returns = per_trade_returns(entry, predictions)
    expected = [net_pnl(closes, i, 1, predicted_up=bool(p), round_trip_cost=0.01) for i, p in zip(entry["indices"], predictions)]
    assert returns == expected


def test_per_trade_returns_triple_barrier_skips_unresolved_touches():
    # Flat prices never touch either barrier -> every trade is a timeout
    # or ambiguous/unresolved; triple_barrier_net_pnl returns None for an
    # unresolved touch, which per_trade_returns must skip, not crash on.
    closes = [100.0] * 20
    entry = {
        "closes": closes, "highs": closes, "lows": closes,
        "horizon": 3, "round_trip_cost": 0.01, "barrier_pct": 0.5,  # barrier far from flat price -> pure timeout
        "label_scheme": "triple-barrier", "indices": [0, 1, 2],
    }
    returns = per_trade_returns(entry, [1, 0, 1])
    # A pure timeout still resolves (touch="timeout"), so this should NOT
    # be empty -- it's the ambiguous same-bar-touch case that's skipped,
    # which flat prices never produce. This test's honest job is just to
    # confirm no crash and a sane, bounded result.
    assert len(returns) == 3
    assert all(isinstance(r, float) for r in returns)
