import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_model import triple_barrier_net_pnl
from strategy.paper import LONG, SHORT, Bar, closed_bars, direction_from_prediction, mean_and_ci, resolve_trade

COST = 0.017
B = 0.03  # barrier


def bars(*hlc, start=1000):
    return [Bar(start + 3600 * k, h, l, c) for k, (h, l, c) in enumerate(hlc)]


def test_long_hits_target():
    r = resolve_trade(100.0, LONG, B, 4, COST, bars((101, 99, 100), (104, 100, 103)))
    assert r.exit_reason == "target" and r.holding_bars == 2
    assert r.gross_return == pytest.approx(B)
    assert r.net_return == pytest.approx(B - COST)


def test_short_hits_target_on_lower_barrier():
    r = resolve_trade(100.0, SHORT, B, 4, COST, bars((101, 96, 97)))
    assert r.exit_reason == "target"
    assert r.gross_return == pytest.approx(B)


def test_long_stopped_and_short_stopped():
    down = bars((101, 96, 97))
    assert resolve_trade(100.0, LONG, B, 4, COST, down).exit_reason == "stop"
    assert resolve_trade(100.0, LONG, B, 4, COST, down).gross_return == pytest.approx(-B)
    up = bars((104, 100, 103))
    assert resolve_trade(100.0, SHORT, B, 4, COST, up).exit_reason == "stop"


def test_timeout_is_kept_and_marked_at_horizon_close():
    r = resolve_trade(100.0, LONG, B, 3, COST, bars((101, 99, 100.5), (101, 99, 101), (101.5, 99.5, 101.2), (200, 1, 100)))
    assert r.exit_reason == "timeout" and r.holding_bars == 3
    assert r.gross_return == pytest.approx(0.012)


def test_open_until_horizon_or_touch():
    assert resolve_trade(100.0, LONG, B, 4, COST, bars((101, 99, 100), (101, 99, 100))) is None
    assert resolve_trade(100.0, LONG, B, 4, COST, []) is None


def test_ambiguous_bar_charged_as_adverse_for_each_direction():
    both = bars((104, 96, 100))
    long_r = resolve_trade(100.0, LONG, B, 4, COST, both)
    short_r = resolve_trade(100.0, SHORT, B, 4, COST, both)
    assert long_r.ambiguous and long_r.exit_reason == "ambiguous_stop" and long_r.gross_return == pytest.approx(-B)
    assert short_r.ambiguous and short_r.gross_return == pytest.approx(-B)


def test_matches_backtest_function_when_resolvable():
    hs = [100, 101, 104, 103, 102]
    ls = [100, 99, 100, 101, 100]
    cs = [100, 100, 103, 102, 101]
    for up in (True, False):
        expected = triple_barrier_net_pnl(hs, ls, cs, 0, 4, B, up, COST)
        r = resolve_trade(
            cs[0], direction_from_prediction(int(up)), B, 4, COST,
            [Bar(k, hs[k], ls[k], cs[k]) for k in range(1, 5)],
        )
        assert r.net_return == pytest.approx(expected)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        resolve_trade(0.0, LONG, B, 4, COST, [])
    with pytest.raises(ValueError):
        resolve_trade(100.0, 0, B, 4, COST, [])


def test_closed_bars_drops_forming_candle():
    raw = [[0, 1, 1, 1, 1], [3600, 1, 1, 1, 1], [7200, 1, 1, 1, 1]]
    assert [c[0] for c in closed_bars(raw, 3600, now=7200 + 10)] == [0, 3600]
    assert [c[0] for c in closed_bars(raw, 3600, now=10800)] == [0, 3600, 7200]


def test_mean_and_ci():
    assert mean_and_ci([]) == (0.0, 0.0, 0)
    assert mean_and_ci([0.5]) == (0.5, 0.0, 1)
    m, hw, n = mean_and_ci([1.0, 3.0])
    assert m == 2.0 and n == 2 and hw > 0
