import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import regime_backtest as rb  # noqa: E402


def series(kind, n=400, seed=1):
    rng = np.random.default_rng(seed)
    if kind == "trend":
        c = 100 * np.exp(np.cumsum(0.004 + 0.002 * rng.standard_normal(n)))
    else:
        c = 100 + 2 * np.sin(np.arange(n) / 3.0) + 0.2 * rng.standard_normal(n)
    return c * 1.004, c * 0.996, c


def test_adx_separates_trend_from_chop():
    h, l, c = series("trend")
    adx, pdi, mdi = rb.adx_di(h, l, c)
    assert np.nanmedian(adx[100:]) > 25 and np.nanmedian(pdi[100:] - mdi[100:]) > 0
    h, l, c = series("range")
    adx, _, _ = rb.adx_di(h, l, c)
    assert np.nanmedian(adx[100:]) < 25


def test_targets_and_stop_are_net_of_fees():
    g, d = rb.gross_target(0.01), rb.stop_drop(0.02)
    net_up = (1 + g) * (1 - rb.FEE) - (1 + rb.FEE)
    assert abs(net_up / (1 + rb.FEE) - 0.01) < 1e-9
    net_dn = (1 - d) * (1 - rb.FEE) - (1 + rb.FEE)
    assert abs(net_dn / (1 + rb.FEE) + 0.02) < 1e-9


def test_a_bar_touching_target_and_stop_counts_as_the_stop():
    n = 40
    c = np.full(n, 100.0)
    h, l = c.copy(), c.copy()
    h[5], l[5] = 110.0, 90.0  # one wild bar after the entry at bar 3
    ix = rb.indicators(h, l, c)
    j, r, why = rb.exit_range_target(0.01)(ix, 3, n)
    assert why == "stop" and j == 5 and r < 0


def test_no_trend_entries_unless_regime_allows():
    h, l, c = series("trend")
    ix = rb.indicators(h, l, c)
    none = np.zeros(len(c), bool)
    assert not rb.entries_trend(ix, none).any()
    up, dn, rg = rb.regimes(ix, "ema")
    assert not (up & dn).any() and not (up & rg).any()
