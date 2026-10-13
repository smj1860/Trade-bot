import sys
from decimal import Decimal as D
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy.exits import exit_reason, stop_price, target_price, tier_plan, trend_intact  # noqa: E402

F = D("0.004")
TIERS = [[0.015, 0.30], [0.025, 0.20], [0.04, 0.25]]


def net_long(entry, exit_, f=F):
    return (exit_ * (1 - f) - entry * (1 + f)) / (entry * (1 + f))


def test_targets_and_stops_are_net_of_fees():
    e = D(30000)
    for net in ("0.015", "0.025", "0.04"):
        p = target_price("BUY", e, D(net), F, F, 4)
        assert net_long(e, p) >= D(net) and net_long(e, p) - D(net) < D("0.0001")
    s = stop_price("BUY", e, D("0.02"), F, F)
    assert abs(net_long(e, s) + D("0.02")) < D("1e-9")
    assert s > e * D("0.985")  # only ~1.2% of price movement, fees take the rest
    short_stop = stop_price("SELL", e, D("0.02"), F, F)
    assert short_stop > e and short_stop < e * D("1.015")


def test_tier_plan_splits_30_20_25_with_a_runner_only_in_an_uptrend():
    plan = tier_plan("BUY", D(30000), D("0.01"), True, TIERS, D("0.015"), F, F, 1, 8)
    assert [q for _, q, _ in plan] == [D("0.003"), D("0.002"), D("0.0025")]
    assert [i for _, _, i in plan] == [0, 1, 2]
    assert plan[0][0] < plan[1][0] < plan[2][0]
    assert sum(q for _, q, _ in plan) == D("0.0075")  # the other 25% is the runner


def test_no_uptrend_means_one_full_size_target():
    plan = tier_plan("BUY", D(30000), D("0.01"), False, TIERS, D("0.015"), F, F, 1, 8)
    assert len(plan) == 1 and plan[0][1] == D("0.01") and plan[0][2] == -1


def test_tiers_round_down_and_tiny_ones_are_dropped():
    plan = tier_plan("BUY", D(100), D("0.00000003"), True, TIERS, D("0.015"), F, F, 2, 8)
    # 30% of 3e-8 -> 0.9e-8 rounds to 0; 20% -> 0; 25% -> 0: nothing left to rest
    assert plan == []
    plan = tier_plan("BUY", D(100), D("0.00000010"), True, TIERS, D("0.015"), F, F, 2, 8)
    assert [q for _, q, _ in plan] == [D("0.00000003"), D("0.00000002"), D("0.00000002")]


def kw(**o):
    d = dict(stop_net=D("0.02"), hard_net=D("0.05"), hold_in_trend=True, fee_entry=F, fee_exit=F)
    d.update(o)
    return d


def test_dip_past_the_stop_is_held_while_the_trend_is_intact():
    e = D(30000)
    below_stop = stop_price("BUY", e, D("0.02"), F, F) - 10
    assert exit_reason("BUY", e, below_stop, 0.01, 0, **kw()) is None  # uptrend: hold
    assert exit_reason("BUY", e, below_stop, -0.01, 0, **kw()) == "stop_trend_broken"
    assert exit_reason("BUY", e, below_stop, 0.0, 0, **kw()) == "stop_trend_broken"  # no history = no reason to hold
    assert exit_reason("BUY", e, below_stop, 0.01, 0, **kw(hold_in_trend=False)) == "stop"


def test_hard_stop_ignores_the_trend():
    e = D(30000)
    way_down = stop_price("BUY", e, D("0.05"), F, F) - 1
    assert exit_reason("BUY", e, way_down, 0.05, 0, **kw()) == "hard_stop"


def test_runner_exits_on_trend_break_only_after_a_tier_filled():
    e = D(30000)
    mid = e * D("1.02")
    assert exit_reason("BUY", e, mid, -0.01, 0, **kw()) is None
    assert exit_reason("BUY", e, mid, -0.01, 1, **kw()) == "runner_trend_exit"
    assert exit_reason("BUY", e, mid, 0.01, 1, **kw()) is None


def test_short_mirrors_long():
    e = D(30000)
    above_stop = stop_price("SELL", e, D("0.02"), F, F) + 10
    assert trend_intact("SELL", -0.01) and not trend_intact("SELL", 0.01)
    assert exit_reason("SELL", e, above_stop, -0.01, 0, **kw()) is None
    assert exit_reason("SELL", e, above_stop, 0.01, 0, **kw()) == "stop_trend_broken"
