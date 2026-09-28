from decimal import Decimal

from strategy.features import Features
from strategy.policy import DecisionPolicy, post_only_price


def make_features(symbol: str = "BTC-USD") -> Features:
    return Features(symbol=symbol, mid_price=Decimal(100), spread=Decimal("0.5"), imbalance=0.0, momentum=0.0)


def make_policy(**overrides) -> DecisionPolicy:
    defaults = dict(
        signal_threshold=0.3,
        cooldown_seconds=10.0,
        order_quantity={"BTC-USD": Decimal("0.01")},
        max_position={"BTC-USD": Decimal("0.05")},
    )
    defaults.update(overrides)
    return DecisionPolicy(**defaults)


def test_below_threshold_no_trade():
    policy = make_policy()
    intent = policy.decide(make_features(), signal=0.2, current_position=Decimal(0), now=0.0)
    assert intent is None


def test_above_threshold_buys():
    policy = make_policy()
    intent = policy.decide(make_features(), signal=0.5, current_position=Decimal(0), now=0.0)
    assert intent is not None
    assert intent.side == "BUY"
    assert intent.quantity == Decimal("0.01")


def test_negative_signal_sells():
    policy = make_policy()
    intent = policy.decide(make_features(), signal=-0.5, current_position=Decimal(0), now=0.0)
    assert intent is not None
    assert intent.side == "SELL"


def test_cooldown_blocks_immediate_repeat():
    policy = make_policy(cooldown_seconds=10.0)
    first = policy.decide(make_features(), signal=0.5, current_position=Decimal(0), now=0.0)
    assert first is not None
    second = policy.decide(make_features(), signal=0.5, current_position=Decimal(0), now=5.0)
    assert second is None


def test_cooldown_expires():
    policy = make_policy(cooldown_seconds=10.0)
    first = policy.decide(make_features(), signal=0.5, current_position=Decimal(0), now=0.0)
    assert first is not None
    second = policy.decide(make_features(), signal=0.5, current_position=Decimal(0), now=11.0)
    assert second is not None


def test_cooldown_is_per_symbol():
    policy = make_policy(
        order_quantity={"BTC-USD": Decimal("0.01"), "ETH-USD": Decimal("0.1")},
        max_position={"BTC-USD": Decimal("0.05"), "ETH-USD": Decimal("0.5")},
    )
    btc = policy.decide(make_features("BTC-USD"), signal=0.5, current_position=Decimal(0), now=0.0)
    eth = policy.decide(make_features("ETH-USD"), signal=0.5, current_position=Decimal(0), now=0.1)
    assert btc is not None
    assert eth is not None


def test_soft_position_limit_blocks_adding_further_in_same_direction():
    policy = make_policy(max_position={"BTC-USD": Decimal("0.05")})
    # Already long 0.045; buying another 0.01 would push to 0.055 > 0.05 limit.
    intent = policy.decide(make_features(), signal=0.5, current_position=Decimal("0.045"), now=0.0)
    assert intent is None


def test_soft_position_limit_allows_reducing_position():
    policy = make_policy(max_position={"BTC-USD": Decimal("0.05")})
    # Long 0.045 (at/near the limit); a SELL reduces exposure, should be allowed.
    intent = policy.decide(make_features(), signal=-0.5, current_position=Decimal("0.045"), now=0.0)
    assert intent is not None
    assert intent.side == "SELL"


def test_no_max_position_configured_for_symbol_means_no_soft_limit():
    policy = make_policy(order_quantity={"BTC-USD": Decimal("0.01")}, max_position={})
    intent = policy.decide(make_features(), signal=0.5, current_position=Decimal("1000"), now=0.0)
    assert intent is not None


# --- Institutional audit Phase 1.3: maker/limit order intents ---


def test_post_only_price_buy_is_the_best_bid():
    # mid=100, spread=0.5 -> best_bid=99.75, best_ask=100.25
    assert post_only_price("BUY", Decimal(100), Decimal("0.5")) == Decimal("99.75")


def test_post_only_price_sell_is_the_best_ask():
    assert post_only_price("SELL", Decimal(100), Decimal("0.5")) == Decimal("100.25")


def test_default_policy_builds_a_limit_order_priced_at_the_best_bid_for_a_buy():
    policy = make_policy()  # use_limit_orders defaults to True
    intent = policy.decide(make_features(), signal=0.5, current_position=Decimal(0), now=0.0)
    assert intent is not None
    assert intent.order_type == "LIMIT"
    assert intent.limit_price == Decimal("99.75")


def test_default_policy_builds_a_limit_order_priced_at_the_best_ask_for_a_sell():
    policy = make_policy()
    intent = policy.decide(make_features(), signal=-0.5, current_position=Decimal(0), now=0.0)
    assert intent is not None
    assert intent.order_type == "LIMIT"
    assert intent.limit_price == Decimal("100.25")


def test_use_limit_orders_false_restores_plain_market_orders():
    policy = make_policy(use_limit_orders=False)
    intent = policy.decide(make_features(), signal=0.5, current_position=Decimal(0), now=0.0)
    assert intent is not None
    assert intent.order_type == "MARKET"
    assert intent.limit_price is None
