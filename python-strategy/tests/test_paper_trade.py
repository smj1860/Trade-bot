import argparse
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.paper_trade import decide_signals, prepare, resolve_open
from scripts.register_paper_model import validate_meta
from scripts.train_model import (
    EXTENDED_DEFAULTS,
    FIB_DEFAULTS,
    WINDOW_ARG_NAMES,
    active_feature_order,
    add_dataset_args,
    build_model_meta,
    features_at,
)

H = 3600
WINDOWS = dict(
    sma_window=20, ema_window=12, rsi_window=14, vol_window=20, bar_momentum_window=10, bollinger_window=20,
    bollinger_num_std=2.0, ao_fast_window=5, ao_slow_window=34, macd_fast_window=12, macd_slow_window=26,
    macd_signal_window=9, cci_window=20, williams_r_window=14,
)
EXT = dict(EXTENDED_DEFAULTS)
FIB = {"windows": list(FIB_DEFAULTS["windows"])}


class FakeModel:
    classes_ = [0, 1]

    def __init__(self, up: bool):
        self.up = up
        self.seen = []

    def predict(self, X):
        self.seen.extend(X)
        return [1 if self.up else 0 for _ in X]

    def predict_proba(self, X):
        return [[0.3, 0.7] if self.up else [0.8, 0.2] for _ in X]


def make_meta(symbols=("BTC-USD",)):
    return {
        "feature_order": active_feature_order(EXT, FIB_DEFAULTS),
        "windows": WINDOWS, "extended": EXT, "fib": FIB, "interval_minutes": 60, "horizon": 4,
        "barrier_by_symbol": {s: 0.03 for s in symbols}, "round_trip_cost": 0.017,
        "symbols": list(symbols), "barrier_mode": "fixed", "label_scheme": "triple-barrier",
    }


def make_rows(n=400, start=1_700_000_000 // H * H, seed=1):
    rng = random.Random(seed)
    price, rows = 100.0, []
    for k in range(n):
        o = price
        price *= 1 + rng.uniform(-0.01, 0.01)
        hi, lo = max(o, price) * 1.002, min(o, price) * 0.998
        rows.append([start + k * H, o, hi, lo, price, price, 10.0 + k % 7, 5])
    return rows


def test_signal_uses_same_features_as_training_function():
    rows = make_rows()
    now = rows[-1][0] + H + 120
    model = FakeModel(up=True)
    sigs, notes = decide_signals(model, make_meta(), {"BTC-USD": rows}, {"BTC-USD": 101.0}, now)
    assert not notes and len(sigs) == 1
    s = sigs[0]
    assert s["entry_ts"] == rows[-1][0] and s["direction"] == 1 and s["entry_price"] == rows[-1][4]
    assert s["observed_price"] == 101.0 and s["proba_up"] == pytest.approx(0.7)
    highs = [r[2] for r in rows]; lows = [r[3] for r in rows]; closes = [r[4] for r in rows]
    vols = [r[6] for r in rows]; mids = [(h + l) / 2 for h, l in zip(highs, lows)]
    feats = features_at(closes, mids, highs, lows, vols, len(rows) - 1, **WINDOWS, extended=EXT, fib=FIB)
    assert model.seen[0] == [feats[n] for n in make_meta()["feature_order"]]


def test_short_signal_direction():
    rows = make_rows()
    sigs, _ = decide_signals(FakeModel(up=False), make_meta(), {"BTC-USD": rows}, {}, rows[-1][0] + H + 5)
    assert sigs[0]["direction"] == -1 and sigs[0]["observed_price"] is None


def test_skips_stale_short_and_unknown_symbols():
    rows = make_rows()
    stale_now = rows[-1][0] + 3 * H
    sigs, notes = decide_signals(FakeModel(True), make_meta(), {"BTC-USD": rows}, {}, stale_now)
    assert not sigs and "stale" in notes[0]
    sigs, notes = decide_signals(FakeModel(True), make_meta(), {"BTC-USD": rows[:50]}, {}, rows[49][0] + H + 5)
    assert not sigs and "need" in notes[0]
    sigs, notes = decide_signals(FakeModel(True), make_meta(), {"ETH-USD": rows}, {}, rows[-1][0] + H + 5)
    assert not sigs and "no barrier" in notes[0]


def test_prepare_drops_forming_candle_and_reports_its_price():
    rows = make_rows(5)
    now = rows[-1][0] + 600  # last candle still forming
    closed, forming = prepare(rows, H, now)
    assert len(closed) == 4 and forming == rows[-1][4]
    closed, forming = prepare(rows, H, rows[-1][0] + H)
    assert len(closed) == 5 and forming is None


def test_resolve_open_target_stop_and_still_open():
    entry = 1_700_000_000 // H * H
    def row(k, hi, lo, c):
        return [entry + k * H, c, hi, lo, c, c, 1, 1]
    open_trades = [
        {"symbol": "A", "entry_ts": entry, "direction": 1, "entry_price": 100.0, "barrier_pct": 0.03, "horizon_bars": 4, "round_trip_cost": 0.017},
        {"symbol": "B", "entry_ts": entry, "direction": -1, "entry_price": 100.0, "barrier_pct": 0.03, "horizon_bars": 4, "round_trip_cost": 0.017},
        {"symbol": "C", "entry_ts": entry, "direction": 1, "entry_price": 100.0, "barrier_pct": 0.03, "horizon_bars": 4, "round_trip_cost": 0.017},
    ]
    candles = {
        "A": [row(0, 100, 100, 100), row(1, 104, 100, 103)],
        "B": [row(0, 100, 100, 100), row(1, 104, 100, 103)],
        "C": [row(0, 100, 100, 100), row(1, 101, 99, 100)],
    }
    out = {t["symbol"]: r for t, r in resolve_open(open_trades, candles)}
    assert out["A"].exit_reason == "target" and out["B"].exit_reason == "stop"
    assert "C" not in out


def test_model_meta_roundtrips_into_a_valid_registration():
    parser = argparse.ArgumentParser()
    add_dataset_args(parser)
    parser.add_argument("--kind", default="gboost")
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args(
        ["--symbol", "all", "--horizon", "4", "--label-scheme", "triple-barrier", "--extended-features",
         "--fib-features", "--profit-margin", "0.04"]
    )
    datasets = [{"symbol": "BTC-USD", "barrier_pct": 0.057, "round_trip_cost": 0.017}]
    metrics = {"model_accuracy": 0.6, "baseline_accuracy": 0.5, "model_mean_pnl": -0.004, "model_pnl_count": 10}
    meta = build_model_meta(args, datasets, metrics)
    assert set(WINDOW_ARG_NAMES) == set(meta["windows"])
    assert meta["feature_order"] == active_feature_order(meta["extended"], FIB_DEFAULTS)
    assert meta["barrier_by_symbol"] == {"BTC-USD": 0.057} and meta["interval_minutes"] == 60
    validate_meta(meta)
    bad = dict(meta, barrier_mode="atr")
    with pytest.raises(ValueError):
        validate_meta(bad)
    with pytest.raises(ValueError):
        validate_meta({k: v for k, v in meta.items() if k != "horizon"})


def test_confidence_filter_helpers():
    from scripts.train_model import confidence_table, confidence_trades, confidence_walk_forward

    # 6 bars; entry at i=0 with barrier 3%: bar 1 high touches upper only.
    entry = {
        "symbol": "A", "label_scheme": "triple-barrier", "idx_test": [0, 2], "start": 0, "end": 2,
        "highs": [100, 104, 100, 100, 100, 100], "lows": [100, 100, 99, 99, 99, 99],
        "closes": [100, 100, 100, 100, 100, 100], "horizon": 2, "barrier_pct": 0.03, "round_trip_cost": 0.01,
    }
    trades = confidence_trades([entry], [0.9, 0.55], [1, 1])
    assert trades[0][0] == pytest.approx(0.4) and trades[0][1] == pytest.approx(0.02)
    assert len(trades) == 2 and trades[1][1] == pytest.approx(-0.01)  # timeout flat, minus cost

    acc = {0: [(0.3, 0.02, 1, "A"), (0.01, -0.03, 0, "A")], 1: [(0.3, 0.01, 1, "A"), (0.02, -0.02, 0, "A")]}
    rows = {r["threshold"]: r for r in confidence_table(acc, thresholds=(0.0, 0.1))}
    assert rows[0.0]["n"] == 4 and rows[0.1]["n"] == 2 and rows[0.1]["mean"] == pytest.approx(0.015)
    assert rows[0.1]["folds_positive"] == 2
    wf = confidence_walk_forward(acc, thresholds=(0.0, 0.1), min_trades=1)
    assert wf["chosen"] == {1: 0.1} and wf["n"] == 1 and wf["mean"] == pytest.approx(0.01)
    assert wf["all_n"] == 2


def test_unresolved_rows_and_pnl():
    from scripts.train_model import build_dataset, unresolved_net_pnl

    n = 80
    closes = [100.0] * n
    highs = [100.5] * n
    lows = [99.5] * n
    # bar 62 spikes both ways (ambiguous for entries before it); the rest never touches 3%.
    highs[62], lows[62] = 105.0, 95.0
    mids = [(h + l) / 2 for h, l in zip(highs, lows)]
    vols = [1.0] * n
    args = (closes, mids, highs, lows, vols, 5, 5, 5, 5, 5, 5, 2.0, 3, 5, 3, 5, 3, 5, 5)
    kw = dict(horizon=3, min_move_threshold=0.03, label_scheme="triple-barrier")
    Xr, yr, ir = build_dataset(*args, **kw)
    Xu, yu, iu = build_dataset(*args, unresolved_only=True, **kw)
    assert not set(ir) & set(iu) and set(iu) and all(v == -1 for v in yu)
    assert 59 in iu and 20 in iu  # ambiguous and timeout rows both land in the unresolved set
    assert unresolved_net_pnl(highs, lows, closes, 59, 3, 0.03, True, 0.01) == pytest.approx(-0.04)
    assert unresolved_net_pnl(highs, lows, closes, 20, 3, 0.03, False, 0.01) == pytest.approx(-0.01)
    resolved_i = next(iter(ir), None)
    assert resolved_i is None or unresolved_net_pnl(highs, lows, closes, resolved_i, 3, 0.03, True, 0.01) is None


def test_two_stage_trades_and_variants():
    from scripts.train_model import confidence_table, two_stage_trades, two_stage_variants

    class M:
        def __init__(self, classes, col1):
            self.classes_ = classes
            self.col1 = col1

        def predict_proba(self, X):
            return [[1 - r[self.col1], r[self.col1]] for r in X]

    n = 30
    highs = [100.0] * n; lows = [100.0] * n; closes = [100.0] * n
    highs[6] = 104.0  # entry 5 (long) hits the 3% target next bar
    e = {
        "symbol": "A", "label_scheme": "triple-barrier", "idx_test": [5, 15], "start": 0, "end": 2,
        "highs": highs, "lows": lows, "closes": closes, "horizon": 2, "barrier_pct": 0.03, "round_trip_cost": 0.01,
        "X_u": [[0.1, 0.2]], "idx_u": [10], "u_move": [0],
    }
    X_test = [[0.9, 0.8], [0.9, 0.8]]  # column 0 -> P(up)=0.9 ; column 1 -> P(move)=0.8 for the fakes below
    dir_model, move_model = M([0, 1], 0), M([0, 1], 1)
    recs = two_stage_trades([e], X_test, dir_model, move_model)
    assert len(recs) == 3  # two test rows + one unresolved row inside the window
    resolved = max(recs, key=lambda r: r[1])
    assert resolved[0] == pytest.approx(0.8) and resolved[1] == pytest.approx(0.02) and resolved[2] == 1
    flat = next(r for r in recs if r[5] == 0)
    assert flat[1] == pytest.approx(-0.01)  # timeout: flat price, minus cost
    assert two_stage_variants({0: recs}, 0.45)[0][0][0] == -1.0  # below the direction floor -> excluded
    acc = two_stage_variants({0: recs}, min_direction_confidence=0.35)
    rows = {r["threshold"]: r for r in confidence_table(acc, (0.0, 0.5))}
    assert rows[0.0]["n"] == 3 and rows[0.5]["n"] == 2
