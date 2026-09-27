from strategy.bars import BarAggregator


def test_no_history_returns_empty():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    assert agg.closes("BTC-USD") == []
    assert agg.window("BTC-USD", 5) == []


def test_first_tick_opens_bucket_but_completes_no_bar():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    agg.on_tick("BTC-USD", timestamp=1000.0, price=100.0)
    assert agg.closes("BTC-USD") == []


def test_ticks_within_same_bucket_update_running_close_only():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    agg.on_tick("BTC-USD", timestamp=0.0, price=100.0)
    agg.on_tick("BTC-USD", timestamp=30.0, price=101.0)
    agg.on_tick("BTC-USD", timestamp=50.0, price=102.0)
    # all three fall in the same [0, 60) bucket -> no completed bar yet
    assert agg.closes("BTC-USD") == []


def test_tick_in_next_bucket_completes_previous_bar():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    agg.on_tick("BTC-USD", timestamp=0.0, price=100.0)
    agg.on_tick("BTC-USD", timestamp=30.0, price=105.0)  # last price while still in bucket [0, 60)
    agg.on_tick("BTC-USD", timestamp=60.0, price=110.0)  # moves into the next bucket [60, 120)
    closes = agg.closes("BTC-USD")
    assert len(closes) == 1
    assert closes[0] == 105.0


def test_multiple_completed_bars_are_ordered_oldest_first():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    agg.on_tick("BTC-USD", timestamp=0.0, price=100.0)
    agg.on_tick("BTC-USD", timestamp=60.0, price=110.0)   # completes bar[0]=100
    agg.on_tick("BTC-USD", timestamp=120.0, price=120.0)  # completes bar[1]=110
    agg.on_tick("BTC-USD", timestamp=180.0, price=130.0)  # completes bar[2]=120
    assert agg.closes("BTC-USD") == [100.0, 110.0, 120.0]


def test_max_bars_evicts_oldest():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=2)
    agg.on_tick("BTC-USD", timestamp=0.0, price=100.0)
    agg.on_tick("BTC-USD", timestamp=60.0, price=110.0)
    agg.on_tick("BTC-USD", timestamp=120.0, price=120.0)
    agg.on_tick("BTC-USD", timestamp=180.0, price=130.0)
    assert agg.closes("BTC-USD") == [110.0, 120.0]


def test_out_of_order_tick_is_ignored():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    agg.on_tick("BTC-USD", timestamp=120.0, price=100.0)
    agg.on_tick("BTC-USD", timestamp=180.0, price=110.0)  # completes bar[0]=100
    agg.on_tick("BTC-USD", timestamp=30.0, price=999.0)  # stale tick, earlier bucket
    assert agg.closes("BTC-USD") == [100.0]


def test_symbols_tracked_independently():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    agg.on_tick("BTC-USD", timestamp=0.0, price=100.0)
    agg.on_tick("BTC-USD", timestamp=60.0, price=110.0)
    agg.on_tick("ETH-USD", timestamp=0.0, price=10.0)
    agg.on_tick("ETH-USD", timestamp=60.0, price=11.0)
    assert agg.closes("BTC-USD") == [100.0]
    assert agg.closes("ETH-USD") == [10.0]


def test_window_returns_most_recent_n():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    for i in range(5):
        agg.on_tick("BTC-USD", timestamp=float(i * 60), price=float(100 + i))
    # 5 ticks -> 4 completed bars: [100, 101, 102, 103]
    assert agg.closes("BTC-USD") == [100.0, 101.0, 102.0, 103.0]
    assert agg.window("BTC-USD", 2) == [102.0, 103.0]
    assert agg.window("BTC-USD", 100) == [100.0, 101.0, 102.0, 103.0]
    assert agg.window("BTC-USD", 0) == []


def test_invalid_construction_args_raise():
    import pytest

    with pytest.raises(ValueError):
        BarAggregator(bar_interval_seconds=0, max_bars=10)
    with pytest.raises(ValueError):
        BarAggregator(bar_interval_seconds=60, max_bars=0)


def test_no_history_midpoints_are_empty():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    assert agg.midpoints("BTC-USD") == []
    assert agg.midpoint_window("BTC-USD", 5) == []


def test_midpoint_uses_bucket_high_low_not_just_close():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    # bucket [0, 60): ticks 100, 105, 98 -> high=105, low=98, close=98 (last)
    agg.on_tick("BTC-USD", timestamp=0.0, price=100.0)
    agg.on_tick("BTC-USD", timestamp=20.0, price=105.0)
    agg.on_tick("BTC-USD", timestamp=40.0, price=98.0)
    agg.on_tick("BTC-USD", timestamp=60.0, price=110.0)  # completes the bucket above
    assert agg.closes("BTC-USD") == [98.0]
    assert agg.midpoints("BTC-USD") == [(105.0 + 98.0) / 2]


def test_midpoint_single_tick_bucket_equals_that_price():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    agg.on_tick("BTC-USD", timestamp=0.0, price=100.0)
    agg.on_tick("BTC-USD", timestamp=60.0, price=110.0)  # completes a bucket with just one tick
    assert agg.midpoints("BTC-USD") == [100.0]


def test_midpoint_window_returns_most_recent_n():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    for i in range(5):
        agg.on_tick("BTC-USD", timestamp=float(i * 60), price=float(100 + i))
    assert agg.midpoints("BTC-USD") == [100.0, 101.0, 102.0, 103.0]
    assert agg.midpoint_window("BTC-USD", 2) == [102.0, 103.0]
    assert agg.midpoint_window("BTC-USD", 0) == []


def test_max_bars_evicts_oldest_midpoints_too():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=2)
    for i in range(5):
        agg.on_tick("BTC-USD", timestamp=float(i * 60), price=float(100 + i))
    assert len(agg.midpoints("BTC-USD")) == 2


def test_no_history_highs_and_lows_are_empty():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    assert agg.highs("BTC-USD") == []
    assert agg.lows("BTC-USD") == []
    assert agg.high_window("BTC-USD", 5) == []
    assert agg.low_window("BTC-USD", 5) == []


def test_highs_and_lows_track_bucket_extremes():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    # bucket [0, 60): ticks 100, 105, 98 -> high=105, low=98
    agg.on_tick("BTC-USD", timestamp=0.0, price=100.0)
    agg.on_tick("BTC-USD", timestamp=20.0, price=105.0)
    agg.on_tick("BTC-USD", timestamp=40.0, price=98.0)
    agg.on_tick("BTC-USD", timestamp=60.0, price=110.0)  # completes the bucket above
    assert agg.highs("BTC-USD") == [105.0]
    assert agg.lows("BTC-USD") == [98.0]


def test_highs_and_lows_window_returns_most_recent_n():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    for i in range(5):
        agg.on_tick("BTC-USD", timestamp=float(i * 60), price=float(100 + i))
    assert agg.highs("BTC-USD") == [100.0, 101.0, 102.0, 103.0]
    assert agg.lows("BTC-USD") == [100.0, 101.0, 102.0, 103.0]
    assert agg.high_window("BTC-USD", 2) == [102.0, 103.0]
    assert agg.low_window("BTC-USD", 2) == [102.0, 103.0]
    assert agg.high_window("BTC-USD", 0) == []
    assert agg.low_window("BTC-USD", 0) == []


def test_max_bars_evicts_oldest_highs_and_lows_too():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=2)
    for i in range(5):
        agg.on_tick("BTC-USD", timestamp=float(i * 60), price=float(100 + i))
    assert len(agg.highs("BTC-USD")) == 2
    assert len(agg.lows("BTC-USD")) == 2


def test_on_trade_first_trade_opens_bucket_but_completes_no_bar():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    agg.on_trade("BTC-USD", timestamp=1000.0, price=100.0, volume=1.0)
    assert agg.closes("BTC-USD") == []
    assert agg.volumes("BTC-USD") == []


def test_on_trade_completes_bar_with_real_close_high_low():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    # bucket [0, 60): trades at 100, 105, 98 -> high=105, low=98, close=98 (last)
    agg.on_trade("BTC-USD", timestamp=0.0, price=100.0, volume=1.0)
    agg.on_trade("BTC-USD", timestamp=20.0, price=105.0, volume=2.0)
    agg.on_trade("BTC-USD", timestamp=40.0, price=98.0, volume=1.0)
    agg.on_trade("BTC-USD", timestamp=60.0, price=110.0, volume=1.0)  # completes the bucket above
    assert agg.closes("BTC-USD") == [98.0]
    assert agg.highs("BTC-USD") == [105.0]
    assert agg.lows("BTC-USD") == [98.0]


def test_on_trade_midpoint_is_real_volume_weighted_vwap_not_high_low_average():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    # bucket [0, 60): (price=100, vol=1) and (price=200, vol=3)
    # vwap = (100*1 + 200*3) / (1 + 3) = 700 / 4 = 175, NOT (100+200)/2 = 150
    agg.on_trade("BTC-USD", timestamp=0.0, price=100.0, volume=1.0)
    agg.on_trade("BTC-USD", timestamp=20.0, price=200.0, volume=3.0)
    agg.on_trade("BTC-USD", timestamp=60.0, price=999.0, volume=1.0)  # completes the bucket above
    assert agg.midpoints("BTC-USD") == [175.0]


def test_on_trade_accumulates_real_volume_per_bar():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    agg.on_trade("BTC-USD", timestamp=0.0, price=100.0, volume=1.5)
    agg.on_trade("BTC-USD", timestamp=20.0, price=101.0, volume=2.5)
    agg.on_trade("BTC-USD", timestamp=60.0, price=102.0, volume=1.0)  # completes the bucket above
    assert agg.volumes("BTC-USD") == [4.0]
    assert agg.volume_window("BTC-USD", 1) == [4.0]


def test_on_trade_out_of_order_trade_is_ignored():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    agg.on_trade("BTC-USD", timestamp=120.0, price=100.0, volume=1.0)
    agg.on_trade("BTC-USD", timestamp=180.0, price=110.0, volume=1.0)  # completes bar[0]=100
    agg.on_trade("BTC-USD", timestamp=30.0, price=999.0, volume=99.0)  # stale, earlier bucket
    assert agg.closes("BTC-USD") == [100.0]


def test_on_trade_single_trade_bucket_vwap_equals_that_price():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    agg.on_trade("BTC-USD", timestamp=0.0, price=100.0, volume=5.0)
    agg.on_trade("BTC-USD", timestamp=60.0, price=110.0, volume=1.0)  # completes the bucket above
    assert agg.midpoints("BTC-USD") == [100.0]


def test_bucket_with_any_real_trade_uses_vwap_even_if_ticks_also_arrived():
    # A bucket that saw both on_tick and on_trade calls uses real VWAP for
    # its midpoint (not the tick-range average) the moment any real trade
    # landed in it — see _complete_bucket's "any real volume" rule.
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    agg.on_tick("BTC-USD", timestamp=0.0, price=100.0)     # tick, no volume
    agg.on_trade("BTC-USD", timestamp=20.0, price=200.0, volume=2.0)  # real trade
    agg.on_trade("BTC-USD", timestamp=60.0, price=999.0, volume=1.0)  # completes the bucket above
    # vwap over the whole bucket = 200*2 / 2 = 200 (the tick contributed no
    # volume, so it doesn't pull the VWAP toward 100)
    assert agg.midpoints("BTC-USD") == [200.0]


def test_tick_only_bucket_still_falls_back_to_high_low_average():
    # A bucket fed only by on_tick (no real trade at all) keeps the
    # pre-existing (high+low)/2 approximation — volume stays 0.0 for it.
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    agg.on_tick("BTC-USD", timestamp=0.0, price=100.0)
    agg.on_tick("BTC-USD", timestamp=20.0, price=105.0)
    agg.on_tick("BTC-USD", timestamp=40.0, price=98.0)
    agg.on_tick("BTC-USD", timestamp=60.0, price=110.0)
    assert agg.midpoints("BTC-USD") == [(105.0 + 98.0) / 2]
    assert agg.volumes("BTC-USD") == [0.0]


def test_on_trade_and_on_tick_track_symbols_independently():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    agg.on_trade("BTC-USD", timestamp=0.0, price=100.0, volume=1.0)
    agg.on_trade("BTC-USD", timestamp=60.0, price=110.0, volume=1.0)
    agg.on_tick("ETH-USD", timestamp=0.0, price=10.0)
    agg.on_tick("ETH-USD", timestamp=60.0, price=11.0)
    assert agg.closes("BTC-USD") == [100.0]
    assert agg.closes("ETH-USD") == [10.0]
    assert agg.volumes("BTC-USD") == [1.0]
    assert agg.volumes("ETH-USD") == [0.0]


def test_max_bars_evicts_oldest_volumes_too():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=2)
    for i in range(5):
        agg.on_trade("BTC-USD", timestamp=float(i * 60), price=float(100 + i), volume=1.0)
    assert len(agg.volumes("BTC-USD")) == 2


def test_no_history_volumes_are_empty():
    agg = BarAggregator(bar_interval_seconds=60, max_bars=10)
    assert agg.volumes("BTC-USD") == []
    assert agg.volume_window("BTC-USD", 5) == []
