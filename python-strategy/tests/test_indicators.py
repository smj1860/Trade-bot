import math

import pytest

from strategy.indicators import (
    _ema_series,
    awesome_oscillator,
    bar_momentum,
    bollinger_bandwidth,
    bollinger_percent_b,
    cci,
    ema,
    ema_ratio,
    macd_histogram,
    realized_vol,
    rsi,
    sma,
    sma_ratio,
    williams_percent_r,
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


def test_ema_series_empty_or_invalid_period_is_empty():
    assert _ema_series([], 12) == []
    assert _ema_series([100.0], 0) == []
    assert _ema_series([100.0], -1) == []


def test_ema_series_single_value_is_seed():
    assert _ema_series([100.0], 12) == [100.0]


def test_ema_series_matches_hand_computed_two_step():
    # period=2 -> alpha = 2/3; seed=100; next = 2/3*110 + 1/3*100
    series = _ema_series([100.0, 110.0], 2)
    assert series[0] == 100.0
    assert abs(series[1] - ((2 / 3) * 110.0 + (1 / 3) * 100.0)) < 1e-9


def test_ema_series_length_matches_input():
    values = [float(x) for x in range(1, 10)]
    assert len(_ema_series(values, 3)) == len(values)


def test_macd_needs_slow_period_of_closes():
    closes = [float(x) for x in range(1, 26)]  # 25 closes, slow_period defaults to 26
    assert macd_histogram(closes) == 0.0


def test_macd_needs_signal_period_of_macd_values_too():
    # The MACD series is as long as the input, so when slow_period is
    # smaller than signal_period, exactly slow_period closes produces
    # fewer MACD values than the signal EMA needs to seed.
    closes = [float(x) for x in range(1, 6)]  # 5 closes == slow_period
    assert macd_histogram(closes, fast_period=2, slow_period=5, signal_period=9) == 0.0


def test_macd_positive_for_steadily_rising_closes():
    # Enough closes to seed slow EMA (26) plus signal EMA (9) = 35, with
    # margin. A steady uptrend should widen fast-EMA-above-slow-EMA
    # faster than the signal line catches up, giving a positive histogram.
    closes = [100.0 + x for x in range(60)]
    result = macd_histogram(closes, fast_period=12, slow_period=26, signal_period=9)
    assert result > 0


def test_macd_zero_for_flat_price():
    closes = [100.0] * 40
    assert macd_histogram(closes, fast_period=12, slow_period=26, signal_period=9) == 0.0


def test_macd_invalid_windows_return_zero():
    closes = [100.0 + x for x in range(60)]
    assert macd_histogram(closes, fast_period=0, slow_period=26, signal_period=9) == 0.0
    assert macd_histogram(closes, fast_period=12, slow_period=0, signal_period=9) == 0.0
    assert macd_histogram(closes, fast_period=12, slow_period=26, signal_period=0) == 0.0


def test_macd_zero_price_is_safe():
    closes = [0.0] * 40
    assert macd_histogram(closes) == 0.0


def test_cci_needs_two_values():
    assert cci([]) == 0.0
    assert cci([100.0]) == 0.0


def test_cci_flat_is_neutral():
    assert cci([100.0] * 20) == 0.0


def test_cci_positive_above_average_negative_below():
    rising_end_high = [100.0, 95.0, 105.0, 90.0, 120.0]
    assert cci(rising_end_high) > 0
    falling_end_low = [100.0, 105.0, 95.0, 110.0, 80.0]
    assert cci(falling_end_low) < 0


def test_williams_r_empty_or_mismatched_is_zero():
    assert williams_percent_r([], [], []) == 0.0
    assert williams_percent_r([100.0], [101.0], []) == 0.0
    assert williams_percent_r([100.0, 101.0], [101.0], [99.0]) == 0.0


def test_williams_r_flat_range_is_zero():
    closes = [100.0] * 5
    highs = [100.0] * 5
    lows = [100.0] * 5
    assert williams_percent_r(closes, highs, lows) == 0.0


def test_williams_r_at_the_high_is_max():
    closes = [100.0, 102.0, 105.0]
    highs = [101.0, 103.0, 105.0]
    lows = [99.0, 101.0, 100.0]
    # most recent close (105) equals the window's highest high (105)
    assert williams_percent_r(closes, highs, lows) == pytest.approx(1.0)


def test_williams_r_at_the_low_is_min():
    closes = [105.0, 102.0, 100.0]
    highs = [106.0, 103.0, 101.0]
    lows = [104.0, 100.0, 100.0]
    # most recent close (100) equals the window's lowest low (100)
    assert williams_percent_r(closes, highs, lows) == pytest.approx(-1.0)
