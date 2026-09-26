import math

from strategy.indicators import (
    awesome_oscillator,
    bar_momentum,
    bollinger_bandwidth,
    bollinger_percent_b,
    ema,
    ema_ratio,
    realized_vol,
    rsi,
    sma,
    sma_ratio,
)


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


def test_ema_empty_is_zero():
    assert ema([]) == 0.0


def test_ema_single_close_is_that_close():
    assert ema([100.0]) == 100.0


def test_ema_matches_hand_computed_two_step():
    # period=2 -> alpha = 2/3; seed=100; next = 2/3*110 + 1/3*100
    closes = [100.0, 110.0]
    expected = (2 / 3) * 110.0 + (1 / 3) * 100.0
    assert abs(ema(closes) - expected) < 1e-9


def test_ema_tracks_flat_price():
    assert ema([100.0, 100.0, 100.0]) == 100.0


def test_ema_ratio_empty_is_zero():
    assert ema_ratio([]) == 0.0


def test_ema_ratio_positive_when_price_above_ema():
    closes = [100.0, 100.0, 100.0, 110.0]
    assert ema_ratio(closes) > 0


def test_ema_ratio_negative_when_price_below_ema():
    closes = [100.0, 100.0, 100.0, 90.0]
    assert ema_ratio(closes) < 0


def test_bollinger_needs_two_closes():
    assert bollinger_percent_b([]) == 0.0
    assert bollinger_percent_b([100.0]) == 0.0
    assert bollinger_bandwidth([]) == 0.0
    assert bollinger_bandwidth([100.0]) == 0.0


def test_bollinger_flat_price_is_neutral_and_zero_width():
    flat = [100.0] * 10
    assert bollinger_percent_b(flat) == 0.0
    assert bollinger_bandwidth(flat) == 0.0


def test_bollinger_percent_b_positive_above_middle_negative_below():
    rising_end_high = [100.0, 95.0, 105.0, 90.0, 120.0]  # last close well above the average
    assert bollinger_percent_b(rising_end_high) > 0
    falling_end_low = [100.0, 105.0, 95.0, 110.0, 80.0]  # last close well below the average
    assert bollinger_percent_b(falling_end_low) < 0


def test_bollinger_bandwidth_positive_for_varying_price():
    varied = [100.0, 110.0, 90.0, 105.0, 95.0]
    assert bollinger_bandwidth(varied) > 0
    assert math.isfinite(bollinger_bandwidth(varied))


def test_awesome_oscillator_needs_full_slow_window():
    assert awesome_oscillator([1.0, 2.0], fast_window=5, slow_window=34) == 0.0
    # exactly slow_window midpoints, all flat -> fast SMA == slow SMA -> 0.0
    flat = [50.0] * 34
    assert awesome_oscillator(flat, fast_window=5, slow_window=34) == 0.0


def test_awesome_oscillator_positive_when_recent_midpoints_rising():
    midpoints = [float(x) for x in range(1, 40)]  # steadily rising, len 39 >= 34
    assert awesome_oscillator(midpoints, fast_window=5, slow_window=34) > 0


def test_awesome_oscillator_negative_when_recent_midpoints_falling():
    midpoints = [float(x) for x in range(39, 0, -1)]  # steadily falling
    assert awesome_oscillator(midpoints, fast_window=5, slow_window=34) < 0


def test_awesome_oscillator_invalid_windows_return_zero():
    midpoints = [float(x) for x in range(1, 40)]
    assert awesome_oscillator(midpoints, fast_window=0, slow_window=34) == 0.0
    assert awesome_oscillator(midpoints, fast_window=5, slow_window=0) == 0.0
