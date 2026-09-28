from decimal import Decimal

from strategy.portfolio import PortfolioManager


def test_starts_at_zero():
    pm = PortfolioManager()
    assert pm.position("BTC-USD") == Decimal(0)


def test_accepted_status_does_not_update_position():
    """ACCEPTED is not a confirmed fill — see portfolio.py's docstring on
    why this only reacts to FILLED/PARTIALLY_FILLED."""
    pm = PortfolioManager()
    pm.on_order_update("co-1", "BTC-USD", "BUY", "ACCEPTED", None)
    assert pm.position("BTC-USD") == Decimal(0)


def test_filled_buy_increases_position():
    pm = PortfolioManager()
    pm.on_order_update("co-1", "BTC-USD", "BUY", "FILLED", Decimal("0.01"))
    assert pm.position("BTC-USD") == Decimal("0.01")


def test_filled_sell_decreases_position():
    pm = PortfolioManager()
    pm.on_order_update("co-1", "BTC-USD", "BUY", "FILLED", Decimal("0.01"))
    pm.on_order_update("co-2", "BTC-USD", "SELL", "FILLED", Decimal("0.004"))
    assert pm.position("BTC-USD") == Decimal("0.006")


def test_partially_filled_updates_position():
    pm = PortfolioManager()
    pm.on_order_update("co-1", "BTC-USD", "BUY", "PARTIALLY_FILLED", Decimal("0.003"))
    assert pm.position("BTC-USD") == Decimal("0.003")


def test_rejected_does_not_update_position():
    pm = PortfolioManager()
    pm.on_order_update("co-1", "BTC-USD", "BUY", "REJECTED", None)
    assert pm.position("BTC-USD") == Decimal(0)


def test_missing_filled_quantity_is_ignored_even_if_status_says_filled():
    pm = PortfolioManager()
    pm.on_order_update("co-1", "BTC-USD", "BUY", "FILLED", None)
    assert pm.position("BTC-USD") == Decimal(0)


def test_symbols_tracked_independently():
    pm = PortfolioManager()
    pm.on_order_update("co-1", "BTC-USD", "BUY", "FILLED", Decimal("0.01"))
    pm.on_order_update("co-2", "ETH-USD", "SELL", "FILLED", Decimal("0.5"))
    assert pm.position("BTC-USD") == Decimal("0.01")
    assert pm.position("ETH-USD") == Decimal("-0.5")


def test_snapshot_excludes_zero_positions():
    pm = PortfolioManager()
    pm.on_order_update("co-1", "BTC-USD", "BUY", "FILLED", Decimal("0.01"))
    pm.on_order_update("co-2", "BTC-USD", "SELL", "FILLED", Decimal("0.01"))  # nets back to zero
    pm.on_order_update("co-3", "ETH-USD", "BUY", "FILLED", Decimal("0.5"))
    snapshot = pm.snapshot()
    assert "BTC-USD" not in snapshot
    assert snapshot["ETH-USD"] == Decimal("0.5")


def test_a_second_update_for_the_same_order_applies_only_the_incremental_delta():
    # Institutional audit Phase 1.3: filled_quantity is CUMULATIVE per
    # order (Kraken's cum_qty — see kraken_private_ws.rs), which matters
    # once a resting limit order can receive more than one update over its
    # lifetime (partial fill, then another partial, then a final fill).
    # Applying each update's filled_quantity directly would double-count.
    pm = PortfolioManager()
    pm.on_order_update("co-1", "BTC-USD", "BUY", "PARTIALLY_FILLED", Decimal("0.003"))
    pm.on_order_update("co-1", "BTC-USD", "BUY", "PARTIALLY_FILLED", Decimal("0.007"))  # cumulative, not +0.007
    pm.on_order_update("co-1", "BTC-USD", "BUY", "FILLED", Decimal("0.01"))  # cumulative, order now fully filled
    assert pm.position("BTC-USD") == Decimal("0.01")


def test_a_duplicate_update_with_no_new_cumulative_fill_is_a_no_op():
    pm = PortfolioManager()
    pm.on_order_update("co-1", "BTC-USD", "BUY", "FILLED", Decimal("0.01"))
    pm.on_order_update("co-1", "BTC-USD", "BUY", "FILLED", Decimal("0.01"))  # same cum_qty repeated
    assert pm.position("BTC-USD") == Decimal("0.01")


def test_multiple_orders_on_the_same_symbol_each_track_their_own_cumulative_fill():
    # Two different orders (different client_order_ids) on the same
    # symbol must not share cumulative-fill state — e.g. a reprice attempt
    # in _manage_resting_order submits a brand-new order for the
    # remaining quantity, which starts its own fill accounting from zero.
    pm = PortfolioManager()
    pm.on_order_update("co-1", "BTC-USD", "BUY", "PARTIALLY_FILLED", Decimal("0.004"))
    pm.on_order_update("co-2", "BTC-USD", "BUY", "FILLED", Decimal("0.006"))
    assert pm.position("BTC-USD") == Decimal("0.01")
