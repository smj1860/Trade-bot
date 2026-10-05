import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_model import limit_entry_trade, parse_fee_scenarios  # noqa: E402

MK, TK, SL, B = 0.004, 0.008, 0.0005, 0.05


def run(highs, lows, closes, up=True, horizon=4):
    return limit_entry_trade(highs, lows, closes, 0, horizon, B, up, MK, TK, SL)


def flat(n=8, price=100.0):
    return [price] * n, [price] * n, [price] * n


def test_unfilled_long_when_next_bar_never_trades_back():
    h, l, c = flat()
    l[1] = 100.5; h[1] = 101.0  # bar 1 entirely above the limit
    assert run(h, l, c) == (None, "unfilled")


def test_target_is_maker_both_legs():
    h, l, c = flat()
    l[1] = 99.9
    h[3] = 100 * (1 + B)
    pnl, reason = run(h, l, c)
    assert reason == "target" and pnl == pytest.approx(B - 2 * MK)


def test_target_in_fill_bar_does_not_count():
    h, l, c = flat()
    l[1] = 99.9; h[1] = 100 * (1 + B) + 1  # fill and target in the same bar
    _, reason = run(h, l, c)
    assert reason == "timeout"


def test_stop_in_fill_bar_counts_and_costs_maker_plus_taker():
    h, l, c = flat()
    l[1] = 100 * (1 - B) - 0.1
    pnl, reason = run(h, l, c)
    assert reason == "stop" and pnl == pytest.approx(-B - (MK + TK + SL))


def test_double_touch_after_fill_is_a_stop():
    h, l, c = flat()
    l[1] = 99.9
    h[2] = 100 * (1 + B); l[2] = 100 * (1 - B)
    assert run(h, l, c)[1] == "stop"


def test_timeout_marks_at_horizon_close_with_taker_exit():
    h, l, c = flat()
    l[1] = 99.9
    c[4] = 101.0
    pnl, reason = run(h, l, c)
    assert reason == "timeout" and pnl == pytest.approx(0.01 - (MK + TK + SL))


def test_short_mirrors_long():
    h, l, c = flat()
    h[1] = 100.2
    l[3] = 100 * (1 - B)
    pnl, reason = run(h, l, c, up=False)
    assert reason == "target" and pnl == pytest.approx(B - 2 * MK)
    h2, l2, c2 = flat()
    h2[1] = 99.5; l2[1] = 99.0  # never rises back to the sell limit
    assert run(h2, l2, c2, up=False) == (None, "unfilled")


def test_no_data_near_end():
    h, l, c = flat(4)
    assert run(h, l, c)[1] == "nodata"


def test_parse_fee_scenarios():
    assert parse_fee_scenarios("0.004:0.008, 0.0022:0.0038") == [(0.004, 0.008), (0.0022, 0.0038)]
    assert parse_fee_scenarios("") == []
