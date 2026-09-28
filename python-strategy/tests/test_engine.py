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
