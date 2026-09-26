"""
Wires everything else in this package into a running process: connects to
the Rust core over gRPC, feeds order book updates into a Strategy,
submits any resulting orders, and logs every step.

Run:
    STRATEGY_CONFIG_PATH=strategy_config.toml python -m strategy.engine
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from typing import Optional

import grpc

from strategy.base import Strategy
from strategy.config import Config
from strategy.imbalance_momentum import ImbalanceMomentumStrategy
from strategy.logging_utils import JsonlEventLogger
from strategy.pb import trading_pb2, trading_pb2_grpc
from strategy.policy import OrderIntent, Side
from strategy.portfolio import PortfolioManager

# Registry of available strategies, keyed by strategy_config.toml's
# `strategy.name`. Adding a new strategy means adding one entry here plus
# the class itself — never touching the engine.
STRATEGIES: dict[str, type[Strategy]] = {
    "imbalance_momentum": ImbalanceMomentumStrategy,
}


def build_strategy(config: Config) -> Strategy:
    cls = STRATEGIES.get(config.strategy.name)
    if cls is None:
        raise ValueError(f"unknown strategy.name: {config.strategy.name!r} (known: {sorted(STRATEGIES)})")
    return cls(config)


class Engine:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.strategy = build_strategy(config)
        self.portfolio = PortfolioManager()
        self.log = JsonlEventLogger(config.logging.log_path, config.logging.level)
        # client_order_id -> (symbol, side), so a later OrderUpdate (which
        # carries no side field of its own — see trading.proto) can still
        # be attributed correctly if/when StreamOrderUpdates actually
        # starts emitting real fills. See portfolio.py's docstring for why
        # this path is currently dormant.
        self._pending_orders: dict[str, tuple[str, Side]] = {}

    async def run(self) -> None:
        async with grpc.aio.insecure_channel(self.config.connection.rust_core_addr) as channel:
            market_stub = trading_pb2_grpc.MarketDataServiceStub(channel)
            order_stub = trading_pb2_grpc.OrderServiceStub(channel)

            self.log.log(
                "startup",
                strategy=self.config.strategy.name,
                symbols=list(self.config.strategy.symbols),
                dry_run_only=self.config.execution.dry_run_only,
            )

            await asyncio.gather(
                self._market_data_loop(market_stub, order_stub),
                self._order_update_loop(order_stub),
            )

    async def _market_data_loop(self, market_stub, order_stub) -> None:
        request = trading_pb2.SubscribeRequest(symbols=list(self.config.strategy.symbols))
        try:
            async for event in market_stub.SubscribeMarketData(request):
                if event.HasField("order_book_update"):
                    await self._handle_order_book_update(event.order_book_update, order_stub)
        except grpc.aio.AioRpcError as e:
            self.log.log("market_data_stream_error", detail=str(e))

    async def _handle_order_book_update(self, update, order_stub) -> None:
        if not update.bids or not update.asks:
            return

        symbol = update.symbol
        best_bid = update.bids[0]
        best_ask = update.asks[0]
        current_position = self.portfolio.position(symbol)

        # exchange_timestamp_ns is the exchange's own event time for this
        # update (see proto/trading.proto); bar-derived features (see
        # strategy/bars.py) bucket on this rather than on when we happened
        # to process the message, so bars line up the same way whether
        # processing is instantaneous or briefly delayed. A zero value
        # (unset) falls back to None, which makes the feature engine use
        # wall-clock time instead.
        timestamp = update.exchange_timestamp_ns / 1_000_000_000 if update.exchange_timestamp_ns else None

        decision = self.strategy.on_order_book_update(
            symbol,
            Decimal(best_bid.price.value),
            Decimal(best_bid.quantity.value),
            Decimal(best_ask.price.value),
            Decimal(best_ask.quantity.value),
            current_position,
            timestamp=timestamp,
        )
        if decision is None:
            return

        self.log.log(
            "signal",
            symbol=symbol,
            mid_price=decision.features.mid_price,
            spread=decision.features.spread,
            imbalance=round(decision.features.imbalance, 4),
            momentum=round(decision.features.momentum, 6),
            sma_ratio=round(decision.features.sma_ratio, 6),
            rsi=round(decision.features.rsi, 4),
            realized_vol=round(decision.features.realized_vol, 6),
            bar_momentum=round(decision.features.bar_momentum, 6),
            signal=round(decision.signal, 4),
            order_side=decision.intent.side if decision.intent else None,
            position=current_position,
        )

        if decision.intent is not None:
            await self._submit_order(decision.intent, order_stub)

    async def _submit_order(self, intent: OrderIntent, order_stub) -> None:
        if self.config.execution.dry_run_only:
            self.log.log(
                "order_would_submit",
                symbol=intent.symbol,
                side=intent.side,
                quantity=intent.quantity,
                signal=round(intent.signal, 4),
                note="execution.dry_run_only=true in strategy_config.toml — never sent to Rust",
            )
            return

        client_order_id = f"{self.strategy.strategy_id}-{intent.symbol}-{time.time_ns()}"
        order = trading_pb2.OrderRequest(
            client_order_id=client_order_id,
            symbol=intent.symbol,
            exchange=self.config.strategy.exchange,
            side=trading_pb2.ORDER_SIDE_BUY if intent.side == "BUY" else trading_pb2.ORDER_SIDE_SELL,
            type=trading_pb2.ORDER_TYPE_MARKET,
            quantity=trading_pb2.Decimal(value=str(intent.quantity)),
            strategy_id=self.strategy.strategy_id,
        )
        self._pending_orders[client_order_id] = (intent.symbol, intent.side)

        self.log.log(
            "order_submitted",
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            signal=round(intent.signal, 4),
        )
        try:
            response = await order_stub.SubmitOrder(order)
        except grpc.aio.AioRpcError as e:
            self.log.log("order_submit_error", symbol=intent.symbol, detail=str(e))
            return

        status_name = trading_pb2.OrderStatus.Name(response.status).removeprefix("ORDER_STATUS_")
        self.log.log(
            "order_result",
            symbol=intent.symbol,
            status=status_name,
            reject_reason=response.reject_reason,
            exchange_order_id=response.exchange_order_id or None,
        )

        filled_qty = _decimal_or_none(response.filled_quantity.value if response.HasField("filled_quantity") else None)
        self.portfolio.on_order_update(intent.symbol, intent.side, status_name, filled_qty)

    async def _order_update_loop(self, order_stub) -> None:
        # Rust's StreamOrderUpdates is still an empty-stream stub
        # (rust-core/src/order.rs) — this will complete almost immediately
        # today with nothing delivered. Wiring it up now means nothing
        # here needs to change once Rust actually streams real fills.
        #
        # One thing that WILL need attention at that point: this loop and
        # _submit_order's synchronous-response handling above both call
        # self.portfolio.on_order_update. Today only the synchronous path
        # ever fires, so there's no double-counting risk — but once real
        # fill streaming exists, the two paths could report the same fill
        # twice unless one of them is removed or the calls are made
        # idempotent (e.g. dedup by exchange_order_id). Flagging this now
        # rather than leaving it to be discovered as a live PnL bug later.
        request = trading_pb2.StreamOrderUpdatesRequest(strategy_id=self.strategy.strategy_id)
        try:
            async for update in order_stub.StreamOrderUpdates(request):
                pending = self._pending_orders.pop(update.client_order_id, None)
                if pending is None:
                    self.log.log(
                        "order_update_unattributed",
                        client_order_id=update.client_order_id,
                        detail="no matching pending order — cannot determine side, skipping portfolio update",
                    )
                    continue
                symbol, side = pending
                status_name = trading_pb2.OrderStatus.Name(update.status).removeprefix("ORDER_STATUS_")
                filled_qty = _decimal_or_none(update.filled_quantity.value if update.HasField("filled_quantity") else None)
                self.portfolio.on_order_update(symbol, side, status_name, filled_qty)
        except grpc.aio.AioRpcError as e:
            self.log.log("order_update_stream_error", detail=str(e))


def _decimal_or_none(value: Optional[str]) -> Optional[Decimal]:
    return Decimal(value) if value else None


def main() -> None:
    config = Config.load()
    engine = Engine(config)
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        pass
    finally:
        engine.log.log("shutdown", positions={k: str(v) for k, v in engine.portfolio.snapshot().items()})
        engine.log.close()


if __name__ == "__main__":
    main()
