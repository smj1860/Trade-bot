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
    assert features.rsi == 0.0
    assert features.realized_vol == 0.0
    assert features.bar_momentum == 0.0


def test_bar_derived_features_populate_once_bars_complete():
    engine = FeatureEngine(
        momentum_window=5,
        bar_interval_seconds=60,
        sma_window=3,
        rsi_window=2,
        vol_window=2,
        bar_momentum_window=2,
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
