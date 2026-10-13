"""
Institutional audit Phase 1.3: covers the maker/limit order lifecycle
state machine in strategy/engine.py's Engine._manage_resting_order —
place -> wait -> cancel/reprice -> fallback-to-market — against a fake
gRPC order stub, per the implementation plan's acceptance criteria (fill-
within-timeout, no-fill-cancel, no-fill-fallback-to-market, partial
fills).

No pytest-asyncio plugin is installed in this project, so each test drives
its own coroutine directly via asyncio.run() rather than using an `async
def test_...` function.
"""

import asyncio
import dataclasses
from decimal import Decimal
from typing import Callable, Optional

from strategy.config import Config
from strategy.engine import Engine
from strategy.pb import trading_pb2
from strategy.policy import OrderIntent


def make_engine(**execution_overrides) -> Engine:
    config = Config.load("strategy_config.example.toml")
    defaults = dict(
        dry_run_only=False,  # the whole point here is exercising real submission logic
        use_limit_orders=True,
        limit_order_timeout_secs=0.05,  # keep "no fill" tests fast
        limit_reprice_attempts=1,
        fallback_to_market=True,
        allow_market_orders=True,  # most tests exercise the (opt-in) market fallback
        take_profit_pct=0.0,  # off unless a test asks for it
    )
    defaults.update(execution_overrides)
    execution = dataclasses.replace(config.execution, **defaults)
    config = dataclasses.replace(config, execution=execution)
    engine = Engine(config)
    # Reprice attempts read the latest known best bid/ask off the engine's
    # order-book cache (populated in production by _handle_order_book_update)
    # to compute a fresh limit price. Tests never feed a real book update,
    # so seed it here to match buy_intent()'s "30000" limit price.
    engine._latest_book["BTC-USD"] = (Decimal("29999"), Decimal("30001"))
    return engine


class FakeOrderStub:
    """A fake OrderServiceStub. `fill_behavior(request) -> Optional[OrderUpdate]`
    decides what (if anything) simulates arriving on the order-update
    stream for a LIMIT SubmitOrder call — see this module's docstring for
    why this is applied synchronously inside SubmitOrder rather than via a
    real background stream: since asyncio is single-threaded/cooperative,
    setting the engine's resting-order event here, before SubmitOrder
    returns, guarantees it's already set by the time
    _manage_resting_order's `await asyncio.wait_for(event.wait(), ...)`
    runs — deterministic, no sleep-based races.
    """

    def __init__(self, engine: Engine, fill_behavior: Callable[[trading_pb2.OrderRequest], Optional[trading_pb2.OrderUpdate]]):
        self.engine = engine
        self.fill_behavior = fill_behavior
        self.submitted: list[trading_pb2.OrderRequest] = []
        self.canceled: list[str] = []

    async def SubmitOrder(self, request: trading_pb2.OrderRequest) -> trading_pb2.OrderUpdate:
        self.submitted.append(request)
        if request.type == trading_pb2.ORDER_TYPE_LIMIT:
            simulated = self.fill_behavior(request)
            if simulated is not None:
                self.engine._resting_order_last_update[request.client_order_id] = simulated
                self.engine._resting_order_events[request.client_order_id].set()
                # Mirrors what the real _order_update_loop does for every
                # update it receives on the stream (including ones for a
                # managed resting order) — _manage_resting_order itself
                # never touches the portfolio directly, so a test bypassing
                # the real stream has to apply this the same way.
                status_name = trading_pb2.OrderStatus.Name(simulated.status).removeprefix("ORDER_STATUS_")
                filled = simulated.filled_quantity.value if simulated.HasField("filled_quantity") else None
                self.engine.portfolio.on_order_update(
                    request.client_order_id,
                    request.symbol,
                    "BUY" if request.side == trading_pb2.ORDER_SIDE_BUY else "SELL",
                    status_name,
                    Decimal(filled) if filled else None,
                )
        return trading_pb2.OrderUpdate(
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            status=trading_pb2.ORDER_STATUS_ACCEPTED,
        )

    async def CancelOrder(self, request: trading_pb2.CancelOrderRequest) -> trading_pb2.CancelOrderResponse:
        self.canceled.append(request.client_order_id)
        return trading_pb2.CancelOrderResponse(outcome=trading_pb2.CANCEL_OUTCOME_CANCELED)


def filled_update(request: trading_pb2.OrderRequest, filled: str) -> trading_pb2.OrderUpdate:
    return trading_pb2.OrderUpdate(
        client_order_id=request.client_order_id,
        symbol=request.symbol,
        status=trading_pb2.ORDER_STATUS_FILLED,
        filled_quantity=trading_pb2.Decimal(value=filled),
    )


def partial_update(request: trading_pb2.OrderRequest, filled: str) -> trading_pb2.OrderUpdate:
    return trading_pb2.OrderUpdate(
        client_order_id=request.client_order_id,
        symbol=request.symbol,
        status=trading_pb2.ORDER_STATUS_PARTIALLY_FILLED,
        filled_quantity=trading_pb2.Decimal(value=filled),
    )


def buy_intent(quantity: str = "0.01", limit_price: str = "30000") -> OrderIntent:
    return OrderIntent(
        symbol="BTC-USD",
        side="BUY",
        quantity=Decimal(quantity),
        signal=0.5,
        order_type="LIMIT",
        limit_price=Decimal(limit_price),
    )


def test_fill_within_timeout_needs_no_cancel_or_reprice():
    engine = make_engine()
    intent = buy_intent("0.01")
    stub = FakeOrderStub(engine, lambda req: filled_update(req, "0.01"))

    asyncio.run(engine._manage_resting_order(intent, stub))

    assert len(stub.submitted) == 1
    assert stub.canceled == []
    assert engine.portfolio.position("BTC-USD") == Decimal("0.01")
    # No leftover bookkeeping for the completed order.
    assert engine._resting_order_events == {}
    assert engine._pending_orders == {}


def test_no_fill_cancels_and_reprices_then_gives_up_without_fallback():
    engine = make_engine(limit_reprice_attempts=1, fallback_to_market=False)
    intent = buy_intent("0.01")
    stub = FakeOrderStub(engine, lambda req: None)  # never fills

    asyncio.run(engine._manage_resting_order(intent, stub))

    # Initial attempt + 1 reprice attempt = 2 submissions, each canceled
    # after its timeout.
    assert len(stub.submitted) == 2
    assert len(stub.canceled) == 2
    assert all(req.type == trading_pb2.ORDER_TYPE_LIMIT for req in stub.submitted)
    assert all(req.post_only for req in stub.submitted)
    # Nothing ever filled, and fallback_to_market=False means no MARKET
    # order was submitted either.
    assert engine.portfolio.position("BTC-USD") == Decimal(0)


def test_no_fill_falls_back_to_market_after_exhausting_reprice_attempts():
    engine = make_engine(limit_reprice_attempts=0, fallback_to_market=True)
    intent = buy_intent("0.01")
    stub = FakeOrderStub(engine, lambda req: None)  # never fills

    asyncio.run(engine._manage_resting_order(intent, stub))

    # One LIMIT attempt (0 reprice attempts configured), then one MARKET
    # fallback order for the full remaining quantity.
    limit_orders = [r for r in stub.submitted if r.type == trading_pb2.ORDER_TYPE_LIMIT]
    market_orders = [r for r in stub.submitted if r.type == trading_pb2.ORDER_TYPE_MARKET]
    assert len(limit_orders) == 1
    assert len(market_orders) == 1
    assert market_orders[0].quantity.value == "0.01"
    assert market_orders[0].symbol == "BTC-USD"
    assert len(stub.canceled) == 1  # the one LIMIT attempt was canceled before falling back


def test_partial_fill_reprices_with_only_the_remaining_quantity():
    engine = make_engine(limit_reprice_attempts=1, fallback_to_market=False)
    intent = buy_intent("0.01")
    calls = {"count": 0}

    def fill_behavior(request):
        calls["count"] += 1
        if calls["count"] == 1:
            return partial_update(request, "0.004")  # half of the first attempt's 0.01
        return filled_update(request, request.quantity.value)  # second attempt fills completely

    stub = FakeOrderStub(engine, fill_behavior)
    asyncio.run(engine._manage_resting_order(intent, stub))

    assert len(stub.submitted) == 2
    assert stub.submitted[0].quantity.value == "0.01"
    # The reprice attempt should ask for only what's left: 0.01 - 0.004.
    assert Decimal(stub.submitted[1].quantity.value) == Decimal("0.006")
    assert len(stub.canceled) == 1  # only the partially-filled first attempt needed canceling
    # Portfolio sees both partial fills, total 0.01 (via distinct
    # client_order_ids — see test_portfolio.py's cumulative-fill tests for
    # why a single order's own cumulative updates don't double count).
    assert engine.portfolio.position("BTC-USD") == Decimal("0.01")


def test_rejected_order_stops_immediately_without_reprice_or_fallback():
    engine = make_engine(limit_reprice_attempts=2, fallback_to_market=True)
    intent = buy_intent("0.01")

    class RejectingStub(FakeOrderStub):
        async def SubmitOrder(self, request):
            self.submitted.append(request)
            return trading_pb2.OrderUpdate(
                client_order_id=request.client_order_id,
                symbol=request.symbol,
                status=trading_pb2.ORDER_STATUS_REJECTED,
                reject_reason="risk engine says no",
            )

    stub = RejectingStub(engine, lambda req: None)
    asyncio.run(engine._manage_resting_order(intent, stub))

    assert len(stub.submitted) == 1  # no retry after a business rejection
    assert stub.canceled == []  # nothing to cancel — it was never accepted
    assert engine.portfolio.position("BTC-USD") == Decimal(0)


def test_market_orders_are_blocked_by_default():
    engine = make_engine(limit_reprice_attempts=0, fallback_to_market=True, allow_market_orders=False)
    stub = FakeOrderStub(engine, lambda req: None)
    asyncio.run(engine._manage_resting_order(buy_intent("0.01"), stub))
    assert [r for r in stub.submitted if r.type == trading_pb2.ORDER_TYPE_MARKET] == []
    assert all(r.post_only for r in stub.submitted)


def test_fill_places_a_post_only_take_profit_netting_1_5_pct_after_fees():
    engine = make_engine(take_profit_pct=0.015, maker_fee_pct=0.004, price_decimals={"BTC-USD": 1})
    stub = FakeOrderStub(engine, lambda req: filled_update(req, "0.01") if req.side == trading_pb2.ORDER_SIDE_BUY else None)
    asyncio.run(engine._manage_resting_order(buy_intent("0.01", "30000"), stub))
    tp = [r for r in stub.submitted if r.side == trading_pb2.ORDER_SIDE_SELL]
    assert len(tp) == 1
    assert tp[0].type == trading_pb2.ORDER_TYPE_LIMIT and tp[0].post_only
    # 30000 * 1.004 * 1.015 / 0.996 = 30694.5.. -> rounded away to 30694.6: about +2.3% from entry
    assert Decimal(tp[0].limit_price.value) == Decimal("30694.6")
    assert Decimal(tp[0].quantity.value) == Decimal("0.01")
    assert tp[0].client_order_id in engine._take_profit_orders


def test_short_entry_takes_profit_below():
    engine = make_engine(take_profit_pct=0.015, maker_fee_pct=0.004, price_decimals={"BTC-USD": 1})
    intent = OrderIntent(symbol="BTC-USD", side="SELL", quantity=Decimal("0.01"), signal=-0.5,
                         order_type="LIMIT", limit_price=Decimal("30000"))
    stub = FakeOrderStub(engine, lambda req: filled_update(req, "0.01") if req.side == trading_pb2.ORDER_SIDE_SELL else None)
    asyncio.run(engine._manage_resting_order(intent, stub))
    tp = [r for r in stub.submitted if r.side == trading_pb2.ORDER_SIDE_BUY]
    assert len(tp) == 1 and Decimal(tp[0].limit_price.value) == Decimal("29314.5")  # 30000*0.996*0.985/1.004 = 29314.5..: about 2.3% below entry


def test_no_take_profit_when_disabled_or_when_the_fill_only_closes_a_position():
    engine = make_engine(take_profit_pct=0.0)
    stub = FakeOrderStub(engine, lambda req: filled_update(req, "0.01"))
    asyncio.run(engine._manage_resting_order(buy_intent("0.01"), stub))
    assert len(stub.submitted) == 1

    engine = make_engine(take_profit_pct=0.015, price_decimals={"BTC-USD": 1})
    engine.portfolio.on_order_update("seed", "BTC-USD", "SELL", "FILLED", Decimal("0.01"))  # already short
    stub = FakeOrderStub(engine, lambda req: filled_update(req, "0.01"))
    asyncio.run(engine._manage_resting_order(buy_intent("0.01"), stub))  # buy closes the short to flat
    assert len(stub.submitted) == 1


def test_take_profit_price_nets_the_target_after_fees_and_rounds_outward():
    from strategy.policy import take_profit_price
    f, net = Decimal("0.004"), Decimal("0.015")
    long_exit = take_profit_price("BUY", Decimal("100"), net, 4, f)
    short_exit = take_profit_price("SELL", Decimal("100"), net, 4, f)
    # round trip after fees on both legs really nets >= 1.5%
    long_net = (long_exit * (1 - f) - 100 * (1 + f)) / (100 * (1 + f))
    short_net = (100 * (1 - f) - short_exit * (1 + f)) / (100 * (1 - f))
    assert long_net >= net and short_net >= net
    assert long_net - net < Decimal("0.0001") and short_net - net < Decimal("0.0001")
    assert long_exit > Decimal("102.3") and short_exit < Decimal("97.8")
    # zero fee reduces to the plain percentage
    assert take_profit_price("BUY", Decimal("100"), net, 2) == Decimal("101.50")


def uptrend_engine(**o):
    engine = make_engine(take_profit_pct=0.015, maker_fee_pct=0.004, price_decimals={"BTC-USD": 1}, **o)
    engine._latest_ema_long_ratio["BTC-USD"] = 0.01  # price above the long EMA
    return engine


def test_uptrend_entry_rests_three_scaled_tiers_and_keeps_a_runner():
    engine = uptrend_engine()
    stub = FakeOrderStub(engine, lambda req: filled_update(req, "0.01") if req.side == trading_pb2.ORDER_SIDE_BUY else None)
    asyncio.run(engine._manage_resting_order(buy_intent("0.01", "30000"), stub))
    tps = [r for r in stub.submitted if r.side == trading_pb2.ORDER_SIDE_SELL]
    assert [Decimal(r.quantity.value) for r in tps] == [Decimal("0.003"), Decimal("0.002"), Decimal("0.0025")]
    prices = [Decimal(r.limit_price.value) for r in tps]
    assert prices == sorted(prices) and prices[0] == Decimal("30694.6")
    assert all(r.post_only and r.type == trading_pb2.ORDER_TYPE_LIMIT for r in tps)
    assert engine._pos["BTC-USD"]["entry"] == Decimal("30000")


def run_exit_check(engine, mid):
    stub = FakeOrderStub(engine, lambda req: filled_update(req, str(req.quantity.value)))

    async def go():
        engine._maybe_start_exit("BTC-USD", Decimal(mid), stub)
        if engine._background_tasks:
            await asyncio.gather(*list(engine._background_tasks))

    asyncio.run(go())
    return stub


def long_position(engine):
    engine.portfolio.on_order_update("seed", "BTC-USD", "BUY", "FILLED", Decimal("0.01"))
    engine._pos["BTC-USD"] = {"side": "BUY", "entry": Decimal("30000"), "tiers_filled": 0, "exiting": False}
    engine._latest_book["BTC-USD"] = (Decimal("29000"), Decimal("29001"))


def test_stop_is_held_through_a_dip_while_the_trend_is_intact():
    engine = uptrend_engine()
    long_position(engine)
    stub = run_exit_check(engine, "29600")  # below the ~29636 net-2% stop
    assert stub.submitted == []
    assert engine._pos["BTC-USD"]["exiting"] is False


def test_stop_sells_post_only_when_the_trend_breaks():
    engine = uptrend_engine()
    long_position(engine)
    engine._latest_ema_long_ratio["BTC-USD"] = -0.002
    stub = run_exit_check(engine, "29600")
    sells = [r for r in stub.submitted if r.side == trading_pb2.ORDER_SIDE_SELL]
    assert len(sells) == 1 and sells[0].type == trading_pb2.ORDER_TYPE_LIMIT and sells[0].post_only
    assert Decimal(sells[0].quantity.value) == Decimal("0.01")
    assert [r for r in stub.submitted if r.type == trading_pb2.ORDER_TYPE_MARKET] == []


def test_hard_stop_exits_even_in_an_uptrend_and_cancels_resting_targets():
    engine = uptrend_engine()
    long_position(engine)
    engine._take_profit_orders["tp-1"] = "BTC-USD"
    engine._resting_order_events["tp-1"] = asyncio.Event()
    stub = run_exit_check(engine, "28700")  # below the net-5% hard stop
    assert "tp-1" in stub.canceled and "tp-1" not in engine._take_profit_orders
    assert any(r.side == trading_pb2.ORDER_SIDE_SELL for r in stub.submitted)
