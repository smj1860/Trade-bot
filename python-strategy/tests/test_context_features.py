import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.register_paper_model import validate_meta  # noqa: E402
from scripts.train_model import (  # noqa: E402
    CONTEXT_DEFAULTS,
    CONTEXT_FEATURES,
    EXTENDED_DEFAULTS,
    FIB_DEFAULTS,
    active_feature_order,
    context_blocks_needed,
    dataset_warmup,
    features_at,
)
from strategy.indicators import daily_context, rolling_blocks  # noqa: E402

WINDOWS = dict(
    sma_window=20, ema_window=12, rsi_window=14, vol_window=20, bar_momentum_window=10, bollinger_window=20,
    bollinger_num_std=2.0, ao_fast_window=5, ao_slow_window=34, macd_fast_window=12, macd_slow_window=26,
    macd_signal_window=9, cci_window=20, williams_r_window=14,
)
CTX = dict(CONTEXT_DEFAULTS, factor=24)


def series(n, seed=3):
    rng = random.Random(seed)
    p, closes, highs, lows, vols = 100.0, [], [], [], []
    for _ in range(n):
        o = p
        p *= 1 + rng.uniform(-0.01, 0.01)
        highs.append(max(o, p) * 1.002)
        lows.append(min(o, p) * 0.998)
        closes.append(p)
        vols.append(10.0)
    mids = [(h + l) / 2 for h, l in zip(highs, lows)]
    return closes, mids, highs, lows, vols


def test_rolling_blocks_end_at_latest_bar():
    closes = list(range(1, 11))
    c, h, l = rolling_blocks(closes, closes, closes, 3, 3)
    assert c == [4, 7, 10]
    assert h == [4, 7, 10] and l == [2, 5, 8]


def test_context_is_causal():
    closes, mids, highs, lows, vols = series(1500)
    i = 1400
    base = features_at(closes, mids, highs, lows, vols, i, **WINDOWS, context=CTX)
    c2, h2, l2 = closes[:], highs[:], lows[:]
    for k in range(i + 1, 1500):
        c2[k] *= 3
        h2[k] *= 3
        l2[k] *= 3
    again = features_at(c2, mids, h2, l2, vols, i, **WINDOWS, context=CTX)
    for name in CONTEXT_FEATURES:
        assert base[name] == again[name]


def test_context_features_present_and_ordered():
    closes, mids, highs, lows, vols = series(1500)
    feats = features_at(closes, mids, highs, lows, vols, 1400, **WINDOWS,
                        extended=EXTENDED_DEFAULTS, fib={"windows": FIB_DEFAULTS["windows"]}, context=CTX)
    order = active_feature_order(EXTENDED_DEFAULTS, FIB_DEFAULTS, CTX)
    assert order[-len(CONTEXT_FEATURES):] == CONTEXT_FEATURES
    assert all(n in feats for n in order)
    assert 0.0 <= feats["d_range_pos"] <= 1.0 and -1.0 <= feats["d_rsi"] <= 1.0


def test_short_history_gives_neutral_defaults():
    closes, _, highs, lows, _ = series(30)
    out = daily_context(closes, highs, lows, 24)
    assert out["d_rsi"] == 0.0 and out["d_range_pos"] == 0.5 and out["d_ema_ratio"] == 0.0


def test_warmup_grows_with_context():
    args = [WINDOWS[k] for k in (
        "sma_window", "ema_window", "rsi_window", "vol_window", "bar_momentum_window", "bollinger_window",
        "ao_slow_window", "macd_slow_window", "macd_signal_window", "cci_window", "williams_r_window")]
    base = dataset_warmup(*args)
    with_ctx = dataset_warmup(*args, context=CTX)
    assert with_ctx >= context_blocks_needed(CTX) * 24 > base


def test_validate_meta_rejects_context_models():
    meta = {
        "feature_order": active_feature_order(None, None, CTX), "windows": WINDOWS, "extended": None, "fib": None,
        "context": CTX, "interval_minutes": 60, "horizon": 4, "barrier_by_symbol": {"BTC-USD": 0.03},
        "round_trip_cost": 0.017, "symbols": ["BTC-USD"], "barrier_mode": "fixed", "label_scheme": "triple-barrier",
    }
    with pytest.raises(Exception, match="context"):
        validate_meta(meta)
