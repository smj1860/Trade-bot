import math

from strategy.indicators import bar_momentum, realized_vol, rsi, sma, sma_ratio


def test_sma_empty_is_zero():
    assert sma([]) == 0.0


def test_sma_basic():
    assert sma([1, 2, 3]) == 2.0


def test_sma_ratio_empty_is_zero():
    assert sma_ratio([]) == 0.0


def test_sma_ratio_above_average_is_positive():
    # closes: 100, 100, 100, 103 -> sma = 100.75, ratio = (103/100.75)-1
    closes = [100, 100, 100, 103]
    expected = (103 / (sum(closes) / len(closes))) - 1.0
    assert abs(sma_ratio(closes) - expected) < 1e-9
    assert sma_ratio(closes) > 0


def test_sma_ratio_below_average_is_negative():
    closes = [100, 100, 100, 97]
    assert sma_ratio(closes) < 0


def test_rsi_needs_two_closes():
    assert rsi([]) == 0.0
    assert rsi([100]) == 0.0


def test_rsi_all_gains_is_max():
    assert rsi([100, 101, 102, 103]) == 1.0


def test_rsi_all_losses_is_min():
    assert rsi([103, 102, 101, 100]) == -1.0


def test_rsi_flat_is_neutral():
    assert rsi([100, 100, 100]) == 0.0


def test_realized_vol_needs_three_closes():
    assert realized_vol([]) == 0.0
    assert realized_vol([100, 101]) == 0.0


def test_realized_vol_zero_for_constant_price():
    assert realized_vol([100, 100, 100, 100]) == 0.0


def test_realized_vol_positive_for_varying_price():
    vol = realized_vol([100, 105, 98, 110, 95])
    assert vol > 0
    assert math.isfinite(vol)


def test_bar_momentum_needs_two_closes():
    assert bar_momentum([]) == 0.0
    assert bar_momentum([100]) == 0.0


def test_bar_momentum_positive_move():
    assert bar_momentum([100, 105, 110]) == 0.1


def test_bar_momentum_negative_move():
    assert bar_momentum([110, 105, 100]) == (100 - 110) / 110


def test_bar_momentum_zero_base_is_safe():
    # oldest close is 0 -> would divide by zero; must return 0.0 instead
    assert bar_momentum([0, 5]) == 0.0
