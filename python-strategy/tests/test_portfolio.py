from decimal import Decimal

from strategy.portfolio import PortfolioManager


def test_starts_at_zero():
    pm = PortfolioManager()
    assert pm.position("BTC-USD") == Decimal(0)


def test_accepted_status_does_not_update_position():
    """ACCEPTED is not a confirmed fill — see portfolio.py's docstring on
    why this only reacts to FILLED/PARTIALLY_FILLED."""
    pm = PortfolioManager()
    pm.on_order_update("BTC-USD", "BUY", "ACCEPTED", None)
    assert pm.position("BTC-USD") == Decimal(0)


def test_filled_buy_increases_position():
    pm = PortfolioManager()
    pm.on_order_update("BTC-USD", "BUY", "FILLED", Decimal("0.01"))
    assert pm.position("BTC-USD") == Decimal("0.01")


def test_filled_sell_decreases_position():
    pm = PortfolioManager()
    pm.on_order_update("BTC-USD", "BUY", "FILLED", Decimal("0.01"))
    pm.on_order_update("BTC-USD", "SELL", "FILLED", Decimal("0.004"))
    assert pm.position("BTC-USD") == Decimal("0.006")


def test_partially_filled_updates_position():
    pm = PortfolioManager()
    pm.on_order_update("BTC-USD", "BUY", "PARTIALLY_FILLED", Decimal("0.003"))
    assert pm.position("BTC-USD") == Decimal("0.003")


def test_rejected_does_not_update_position():
    pm = PortfolioManager()
    pm.on_order_update("BTC-USD", "BUY", "REJECTED", None)
    assert pm.position("BTC-USD") == Decimal(0)


def test_missing_filled_quantity_is_ignored_even_if_status_says_filled():
    pm = PortfolioManager()
    pm.on_order_update("BTC-USD", "BUY", "FILLED", None)
    assert pm.position("BTC-USD") == Decimal(0)


def test_symbols_tracked_independently():
    pm = PortfolioManager()
    pm.on_order_update("BTC-USD", "BUY", "FILLED", Decimal("0.01"))
    pm.on_order_update("ETH-USD", "SELL", "FILLED", Decimal("0.5"))
    assert pm.position("BTC-USD") == Decimal("0.01")
    assert pm.position("ETH-USD") == Decimal("-0.5")


def test_snapshot_excludes_zero_positions():
    pm = PortfolioManager()
    pm.on_order_update("BTC-USD", "BUY", "FILLED", Decimal("0.01"))
    pm.on_order_update("BTC-USD", "SELL", "FILLED", Decimal("0.01"))  # nets back to zero
    pm.on_order_update("ETH-USD", "BUY", "FILLED", Decimal("0.5"))
    snapshot = pm.snapshot()
    assert "BTC-USD" not in snapshot
    assert snapshot["ETH-USD"] == Decimal("0.5")
