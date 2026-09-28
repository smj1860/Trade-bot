from decimal import Decimal

from strategy.features import FeatureEngine


def test_returns_none_for_crossed_or_empty_book():
    engine = FeatureEngine(momentum_window=5)
    assert engine.on_order_book_update("BTC-USD", Decimal(0), Decimal(1), Decimal(100), Decimal(1)) is None
    # crossed book: bid >= ask
    assert engine.on_order_book_update("BTC-USD", Decimal(101), Decimal(1), Decimal(100), Decimal(1)) is None


def test_mid_price_and_spread():
    engine = FeatureEngine(momentum_window=5)
    features = engine.on_order_book_update("BTC-USD", Decimal(100), Decimal(2), Decimal(102), Decimal(2))
    assert features is not None
    assert features.mid_price == Decimal(101)
    assert features.spread == Decimal(2)


def test_imbalance_positive_when_more_bid_size():
    engine = FeatureEngine(momentum_window=5)
    features = engine.on_order_book_update("BTC-USD", Decimal(100), Decimal(9), Decimal(102), Decimal(1))
    assert features is not None
    assert features.imbalance == 0.8  # (9-1)/10


def test_imbalance_negative_when_more_ask_size():
    engine = FeatureEngine(momentum_window=5)
    features = engine.on_order_book_update("BTC-USD", Decimal(100), Decimal(1), Decimal(102), Decimal(9))
    assert features is not None
    assert features.imbalance == -0.8


def test_imbalance_zero_when_book_has_no_size():
    engine = FeatureEngine(momentum_window=5)
    features = engine.on_order_book_update("BTC-USD", Decimal(100), Decimal(0), Decimal(102), Decimal(0))
    assert features is not None
    assert features.imbalance == 0.0


def test_momentum_zero_on_first_tick():
    engine = FeatureEngine(momentum_window=3)
    features = engine.on_order_book_update("BTC-USD", Decimal(100), Decimal(1), Decimal(102), Decimal(1))
    assert features.momentum == 0.0


def test_momentum_reflects_price_change_over_window():
    engine = FeatureEngine(momentum_window=2)
    # tick 1: mid = 101
    engine.on_order_book_update("BTC-USD", Decimal(100), Decimal(1), Decimal(102), Decimal(1))
    # tick 2: mid = 103 (history still has only tick 1 as reference)
    engine.on_order_book_update("BTC-USD", Decimal(102), Decimal(1), Decimal(104), Decimal(1))
    # tick 3: window is now full (maxlen=3); reference is still tick 1's mid (101)
    features = engine.on_order_book_update("BTC-USD", Decimal(104), Decimal(1), Decimal(106), Decimal(1))
    assert features is not None
    # mid now 105, reference 101 -> (105-101)/101
    expected = float((Decimal(105) - Decimal(101)) / Decimal(101))
    assert abs(features.momentum - expected) < 1e-9


def test_symbols_tracked_independently():
    engine = FeatureEngine(momentum_window=5)
    btc = engine.on_order_book_update("BTC-USD", Decimal(100), Decimal(1), Decimal(102), Decimal(1))
    eth = engine.on_order_book_update("ETH-USD", Decimal(10), Decimal(1), Decimal(11), Decimal(1))
    assert btc.symbol == "BTC-USD"
    assert eth.symbol == "ETH-USD"
    assert btc.momentum == 0.0
    assert eth.momentum == 0.0


def test_bar_derived_features_default_to_neutral_with_no_bar_history():
    # No timestamps passed in yet (or too little history) -> all bar-derived
    # features should come back at their neutral 0.0 default, matching
    # strategy.indicators' own "not enough history" convention.
    engine = FeatureEngine(momentum_window=5, bar_interval_seconds=60)
    features = engine.on_order_book_update(
        "BTC-USD", Decimal(100), Decimal(1), Decimal(102), Decimal(1), timestamp=0.0
    )
    assert features.sma_ratio == 0.0
    assert features.ema_ratio == 0.0
    assert features.rsi == 0.0
    assert features.realized_vol == 0.0
    assert features.bar_momentum == 0.0
    assert features.bollinger_percent_b == 0.0
    assert features.bollinger_bandwidth == 0.0
    assert features.awesome_oscillator == 0.0
    assert features.macd_histogram == 0.0
    assert features.cci == 0.0
    assert features.williams_percent_r == 0.0
    assert features.volume_ratio == 0.0
    assert features.parkinson_vol == 0.0


def test_bar_derived_features_populate_once_bars_complete():
    engine = FeatureEngine(
        momentum_window=5,
        bar_interval_seconds=60,
        sma_window=3,
        ema_window=3,
        rsi_window=2,
        vol_window=2,
        bar_momentum_window=2,
        bollinger_window=3,
    )
    # Each call's mid-price is (bid+ask)/2 = 100, 101, ..., rising steadily.
    # One tick per 60s bucket -> each call after the first completes a bar
    # with the previous call's mid-price as its close.
    for i in range(5):
        features = engine.on_order_book_update(
            "BTC-USD",
            Decimal(100 + i),
            Decimal(1),
            Decimal(102 + i),
            Decimal(1),
            timestamp=float(i * 60),
        )
    # After 5 ticks there are 4 completed bars: closes 101, 102, 103, 104
    # (mid-price of each of the first four ticks).
    assert features is not None
    assert features.bar_momentum > 0  # steadily rising closes
    assert features.rsi == 1.0  # every step was a gain
    assert features.sma_ratio > 0  # most recent close above the window's average
    assert features.ema_ratio > 0  # most recent close above its EMA
    assert features.bollinger_percent_b > 0  # most recent close above its middle band


def test_awesome_oscillator_populates_once_slow_window_completes():
    engine = FeatureEngine(momentum_window=5, bar_interval_seconds=60, ao_fast_window=2, ao_slow_window=4)
    # Rising mid-prices -> rising bar midpoints -> fast SMA above slow SMA.
    for i in range(6):
        features = engine.on_order_book_update(
            "BTC-USD",
            Decimal(100 + i),
            Decimal(1),
            Decimal(102 + i),
            Decimal(1),
            timestamp=float(i * 60),
        )
    assert features is not None
    assert features.awesome_oscillator > 0


def test_volume_features_stay_neutral_without_a_trade_feed():
    # on_order_book_update alone (no on_trade calls) only ever feeds
    # bars.py's on_tick fallback, which never accumulates real volume —
    # volume_ratio/parkinson_vol should still populate (parkinson_vol from
    # the tick-derived high/low range) or stay neutral (volume_ratio,
    # since real volume is always 0 without a trade feed), never raise.
    engine = FeatureEngine(momentum_window=5, bar_interval_seconds=60, vol_window=2)
    for i in range(4):
        features = engine.on_order_book_update(
            "BTC-USD", Decimal(100 + i), Decimal(1), Decimal(102 + i), Decimal(1), timestamp=float(i * 60)
        )
    assert features is not None
    assert features.volume_ratio == 0.0  # no real trade volume ever reported


def test_volume_ratio_populates_once_a_trade_feed_is_wired_up():
    engine = FeatureEngine(momentum_window=5, bar_interval_seconds=60, vol_window=2)
    # Three quiet-volume bars, then one high-volume bar -> volume_ratio > 0
    # for the next order-book update (which reads the now-completed bars).
    engine.on_trade("BTC-USD", price=Decimal(100), volume=Decimal(10), timestamp=0.0)
    engine.on_trade("BTC-USD", price=Decimal(101), volume=Decimal(10), timestamp=60.0)
    engine.on_trade("BTC-USD", price=Decimal(102), volume=Decimal(10), timestamp=120.0)
    engine.on_trade("BTC-USD", price=Decimal(103), volume=Decimal(100), timestamp=180.0)
    features = engine.on_order_book_update(
        "BTC-USD", Decimal(103), Decimal(1), Decimal(105), Decimal(1), timestamp=240.0
    )
    assert features is not None
    assert features.volume_ratio > 0
    assert features.parkinson_vol >= 0.0


def test_on_trade_symbol_not_yet_seen_by_order_book_update_does_not_raise():
    engine = FeatureEngine(momentum_window=5)
    # on_trade shouldn't require on_order_book_update to have run first.
    engine.on_trade("BTC-USD", price=Decimal(100), volume=Decimal(5), timestamp=0.0)


def test_macd_populates_once_windows_complete():
    engine = FeatureEngine(
        momentum_window=5,
        bar_interval_seconds=60,
        macd_fast_window=2,
        macd_slow_window=4,
        macd_signal_window=2,
    )
    # Steadily rising mid-prices -> steadily rising bar closes -> a
    # positive MACD histogram (fast EMA pulling away above slow EMA).
    for i in range(12):
        features = engine.on_order_book_update(
            "BTC-USD",
            Decimal(100 + i),
            Decimal(1),
            Decimal(102 + i),
            Decimal(1),
            timestamp=float(i * 60),
        )
    assert features is not None
    assert features.macd_histogram > 0


def test_cci_populates_once_window_completes():
    engine = FeatureEngine(momentum_window=5, bar_interval_seconds=60, cci_window=3)
    # Rising mid-prices -> rising typical prices -> most recent above its
    # average -> positive CCI.
    for i in range(6):
        features = engine.on_order_book_update(
            "BTC-USD",
            Decimal(100 + i),
            Decimal(1),
            Decimal(102 + i),
            Decimal(1),
            timestamp=float(i * 60),
        )
    assert features is not None
    assert features.cci > 0


def test_williams_r_populates_once_window_completes():
    engine = FeatureEngine(momentum_window=5, bar_interval_seconds=60, williams_r_window=3)
    # Steadily rising mid-prices -> the most recent close sits at (or near)
    # the window's high -> williams %r close to +1 (bullish).
    for i in range(6):
        features = engine.on_order_book_update(
            "BTC-USD",
            Decimal(100 + i),
            Decimal(1),
            Decimal(102 + i),
            Decimal(1),
            timestamp=float(i * 60),
        )
    assert features is not None
    assert features.williams_percent_r > 0
