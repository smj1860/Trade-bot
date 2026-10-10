import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import rule_backtest as rb  # noqa: E402


def walk(n=3000, drift=0.0, vol=0.006, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    h, l = c * (1 + abs(rng.normal(0, vol / 2, n))), c * (1 - abs(rng.normal(0, vol / 2, n)))
    v = rng.uniform(50, 150, n)
    ts = np.arange(n, dtype=np.int64) * 3600 + 1_700_000_000
    return ts, h, l, c, v


def test_tiers():
    assert [rb.tier_of(x) for x in (0.1, 0.35, 0.5, 0.55, 0.9, -0.8)] == [0, 1, 1, 2, 3, 3]


def test_simulate_exit_long_and_short_and_ambiguity():
    c = np.array([100.0, 100, 100, 100, 100])
    h = np.array([100.0, 100.5, 102.5, 100, 100])
    l = np.array([100.0, 99.8, 99.9, 100, 100])
    assert rb.simulate_exit(h, l, c, 0, 1, 0.02, 0.01, 3) == ("target", 0.02, 2)
    assert rb.simulate_exit(h, l, c, 0, -1, 0.02, 0.01, 3) == ("stop", -0.01, 2)
    # a bar touching both barriers is a stop for the long
    h2, l2 = np.array([100, 103.0, 100, 100]), np.array([100, 98.0, 100, 100])
    assert rb.simulate_exit(h2, l2, np.full(4, 100.0), 0, 1, 0.02, 0.01, 2)[0] == "stop"
    # timeout marks at the vertical barrier close; data ending first gives None
    c3 = np.array([100.0, 100.2, 100.4, 101.0])
    q = rb.simulate_exit(c3, c3, c3, 0, 1, 0.05, 0.05, 3)
    assert q[0] == "timeout" and q[1] == pytest.approx(0.01)
    assert rb.simulate_exit(c3, c3, c3, 0, 1, 0.05, 0.05, 4) is None


def test_score_is_positive_in_an_uptrend_and_negative_in_a_downtrend():
    for drift, sign in ((0.002, 1), (-0.002, -1)):
        ts, h, l, c, v = walk(drift=drift, vol=0.002)
        s = rb.score(rb.features(h, l, c, v), c)
        assert np.nanmean(s[200:]) * sign > 0.3
        assert np.nanmax(np.abs(s[100:])) <= 1.15 + 1e-9


def test_contiguous_flags_drop_windows_around_missing_bars():
    ts = np.arange(100, dtype=np.int64) * 3600
    ts[50:] += 3600  # one missing candle before index 50
    ok = rb.contiguous_flags(ts, 3600, 5, 5)
    assert ok[20] and ok[80] and not ok[48] and not ok[52]


def test_backtest_runs_and_controls_match_entries():
    ts, h, l, c, v = walk(n=4000, drift=0.0003)
    cfg = rb.CONFIGS["A (tp 2/3.5/5%, sl 1/1.5/2%, 48 bars)"]
    trades, entries = rb.backtest_symbol("X", ts, h, l, c, v, cfg)
    assert trades and len(trades) == len(entries)
    assert all(t.tier in (1, 2, 3) and t.reason in ("target", "stop", "timeout") for t in trades)
    signs = [t.direction for t in trades]
    rev = rb.control_trades("X", ts, h, l, c, entries, cfg, "reverse", sign=signs)
    assert [-t.direction for t in rev] == signs[: len(rev)]
    # positions do not overlap
    for a, b in zip(trades, trades[1:]):
        assert b.i >= a.i + a.bars


def test_full_report_text():
    text = rb.run({"X-USD": walk(n=3000, drift=0.0002)}, 3600, 0.001, n_random=2)
    assert "config A" in text and "config B" in text and "reversed direction" in text and "random direction #2" in text


def test_regime_and_min_score_filters_and_daily_holds():
    ts, h, l, c, v = walk(n=5000, drift=0.0003)
    cfg = next(iter(rb.configs_for(60).values()))
    base, _ = rb.backtest_symbol("X", ts, h, l, c, v, cfg)
    strict, _ = rb.backtest_symbol("X", ts, h, l, c, v, cfg, min_score=0.75)
    assert 0 < len(strict) < len(base) and all(abs(t.score) >= 0.75 for t in strict)
    reg, _ = rb.backtest_symbol("X", ts, h, l, c, v, cfg, regime=True)
    f = rb.features(h, l, c, v)
    assert reg and all(f["regime"][t.i] == t.direction for t in reg)
    assert [c_["hold"] for c_ in rb.configs_for(1440).values()] == [20, 40, 20]
    assert [c_["hold"] for c_ in rb.configs_for(240).values()] == [48, 96, 48]
    assert "variant: min |score| 0.75" in rb.run({"X": (ts, h, l, c, v)}, 3600, 0.001, n_random=1, min_score=0.75, regime=True)


def test_macd_cross_trigger_only_enters_on_fresh_crosses():
    ts, h, l, c, v = walk(n=5000, drift=0.0003)
    cfg = next(iter(rb.configs_for(60).values()))
    base, _ = rb.backtest_symbol("X", ts, h, l, c, v, cfg)
    cx, _ = rb.backtest_symbol("X", ts, h, l, c, v, cfg, macd_cross=True)
    f = rb.features(h, l, c, v)
    assert 0 < len(cx) < len(base) and all(f["xdir"][t.i] == t.direction for t in cx)
    # xdir is +1/-1 only within two bars after the histogram changes sign
    hist = f["hist"]
    i = next(k for k in range(60, 4000) if hist[k - 1] <= 0 < hist[k])
    assert f["xdir"][i] == 1 and f["xdir"][i + 2] in (1, -1)
