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
        # be attributed correctly. Real fills now flow through this (see
        # kraken_private_ws.rs), not just a hypothetical future pipeline.
        self._pending_orders: dict[str, tuple[str, Side]] = {}
        # Institutional audit Phase 1.3: a resting limit order's lifecycle
        # (_manage_resting_order) needs to wake up as soon as a fill/cancel
        # update arrives for the specific order it's waiting on, rather
        # than only finding out once its whole timeout elapses. Entries
        # exist only while a resting order is actively being managed;
        # _order_update_loop populates _resting_order_last_update and sets
        # the matching event, and _manage_resting_order (via
        # _cleanup_resting_order) removes both once it's done with that
        # order — see that method's docstring for why this is a separate,
        # additive mechanism rather than a change to _pending_orders'
        # existing pop-once behavior.
        self._resting_order_events: dict[str, asyncio.Event] = {}
        self._resting_order_last_update: dict[str, trading_pb2.OrderUpdate] = {}
        # symbol -> (best_bid, best_ask) from the most recent order book
        # update, read by _manage_resting_order when repricing a limit
        # order at each reprice attempt — the price at decision time is
        # stale by the time a cancel-and-reprice actually happens.
        self._latest_book: dict[str, tuple[Decimal, Decimal]] = {}
        # A resting order's lifecycle runs as a background task (see
        # _submit_order) rather than being awaited inline, so a slow-to-
        # fill limit order never blocks market-data processing for every
        # other symbol. Tracked here purely so a task's exception surfaces
        # (via its done-callback) instead of being silently dropped, and
        # so run() could in principle wait for them to drain on shutdown.
        self._background_tasks: set[asyncio.Task] = set()

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
                self._heartbeat_loop(order_stub),
            )

    async def _heartbeat_loop(self, order_stub) -> None:
        # Institutional audit Phase 1.1: the Rust side's dead-man's switch
        # (rust-core/src/heartbeat.rs) treats this strategy as dead — and
        # cancels its resting orders and/or flattens its positions,
        # depending on config — once it stops seeing these calls for
        # dead_man_switch.heartbeat_timeout_secs. This loop is the other
        # half of that: it must keep running independently of whether the
        # strategy is actually generating signals right now (a quiet
        # market is not a dead process), which is why it's its own
        # asyncio.gather task rather than piggybacked on the market-data
        # loop.
        interval = self.config.connection.heartbeat_interval_secs
        heartbeat = trading_pb2.Heartbeat(strategy_id=self.strategy.strategy_id)
        while True:
            try:
                ack = await order_stub.SendHeartbeat(heartbeat)
                if ack.heartbeat_timeout_secs and interval * 2 > ack.heartbeat_timeout_secs:
                    self.log.log(
                        "heartbeat_interval_too_close_to_timeout",
                        interval_secs=interval,
                        rust_timeout_secs=ack.heartbeat_timeout_secs,
                        note="this process's heartbeat interval is not comfortably under Rust's configured "
                        "dead_man_switch.heartbeat_timeout_secs — a single slow/missed call could trip it",
                    )
            except grpc.aio.AioRpcError as e:
                # A single failed heartbeat isn't fatal — it just means
                # this tick didn't reset Rust's staleness clock. Logged so
                # sustained failures are visible, not raised, since a
                # transient network hiccup here shouldn't crash the whole
                # strategy process (that would be a self-inflicted version
                # of exactly what this loop exists to detect).
                self.log.log("heartbeat_error", detail=str(e))
            await asyncio.sleep(interval)

    async def _market_data_loop(self, market_stub, order_stub) -> None:
        request = trading_pb2.SubscribeRequest(symbols=list(self.config.strategy.symbols))
        try:
            async for event in market_stub.SubscribeMarketData(request):
                if event.HasField("order_book_update"):
                    await self._handle_order_book_update(event.order_book_update, order_stub)
                elif event.HasField("trade_update"):
                    self._handle_trade_update(event.trade_update)
        except grpc.aio.AioRpcError as e:
            self.log.log("market_data_stream_error", detail=str(e))

    def _handle_trade_update(self, update) -> None:
        # Real executed trade (see proto/trading.proto's TradeUpdate and
        # rust-core/src/kraken.rs's trade-channel subscription) — feeds
        # strategy.bars.BarAggregator's real-VWAP path via
        # Strategy.on_trade, closing the live/historical feature-parity
        # gap that on_order_book_update's mid-price-tick fallback left
        # open. Produces no decision/order on its own (see
        # Strategy.on_trade's docstring), so there's nothing to log here
        # beyond what bar completion will surface on the next order book
        # update's "signal" log line.
        timestamp = update.exchange_timestamp_ns / 1_000_000_000 if update.exchange_timestamp_ns else None
        self.strategy.on_trade(
            update.symbol,
            Decimal(update.price.value),
            Decimal(update.quantity.value),
            timestamp=timestamp,
        )

    async def _handle_order_book_update(self, update, order_stub) -> None:
        if not update.bids or not update.asks:
            return

        symbol = update.symbol
        best_bid = update.bids[0]
        best_ask = update.asks[0]
        current_position = self.portfolio.position(symbol)
        # See _manage_resting_order — read at reprice time, not decision
        # time, since the book has likely moved by then.
        self._latest_book[symbol] = (Decimal(best_bid.price.value), Decimal(best_ask.price.value))

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
            ema_ratio=round(decision.features.ema_ratio, 6),
            rsi=round(decision.features.rsi, 4),
            realized_vol=round(decision.features.realized_vol, 6),
            bar_momentum=round(decision.features.bar_momentum, 6),
            bollinger_percent_b=round(decision.features.bollinger_percent_b, 4),
            bollinger_bandwidth=round(decision.features.bollinger_bandwidth, 6),
            awesome_oscillator=round(decision.features.awesome_oscillator, 6),
            macd_histogram=round(decision.features.macd_histogram, 6),
            cci=round(decision.features.cci, 4),
            williams_percent_r=round(decision.features.williams_percent_r, 4),
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
                order_type=intent.order_type,
                limit_price=str(intent.limit_price) if intent.limit_price is not None else None,
                signal=round(intent.signal, 4),
                note="execution.dry_run_only=true in strategy_config.toml — never sent to Rust",
            )
            return

        if intent.order_type == "LIMIT":
            # Institutional audit Phase 1.3: a resting order's lifecycle
            # (place -> wait -> cancel/reprice -> fallback) can take
            # several seconds across multiple attempts — run it as a
            # background task rather than blocking this coroutine (and
            # therefore _market_data_loop, and therefore every other
            # symbol's book processing) for that whole time.
            task = asyncio.create_task(self._manage_resting_order(intent, order_stub))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return

        await self._submit_market_order(intent.symbol, intent.side, intent.quantity, intent.signal, order_stub)

    async def _submit_market_order(
        self, symbol: str, side: Side, quantity: Decimal, signal: float, order_stub
    ) -> Optional[str]:
        """Submits a plain MARKET order and applies its synchronous result
        to the portfolio. Used both for a policy decision that was never a
        LIMIT order in the first place (execution.use_limit_orders=False)
        and for `_manage_resting_order`'s fallback-to-market path once a
        resting order's reprice attempts are exhausted. Returns the
        resulting status name (e.g. "ACCEPTED", "REJECTED"), or None on a
        transport failure, so a caller that cares (currently none do) can
        react without re-parsing a log line."""
        client_order_id = f"{self.strategy.strategy_id}-{symbol}-{time.time_ns()}"
        order = trading_pb2.OrderRequest(
            client_order_id=client_order_id,
            symbol=symbol,
            exchange=self.config.strategy.exchange,
            side=_side_to_pb(side),
            type=trading_pb2.ORDER_TYPE_MARKET,
            quantity=trading_pb2.Decimal(value=str(quantity)),
            strategy_id=self.strategy.strategy_id,
        )
        self._pending_orders[client_order_id] = (symbol, side)

        self.log.log(
            "order_submitted",
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type="MARKET",
            signal=round(signal, 4),
        )
        try:
            response = await order_stub.SubmitOrder(order)
        except grpc.aio.AioRpcError as e:
            self.log.log("order_submit_error", symbol=symbol, detail=str(e))
            self._pending_orders.pop(client_order_id, None)
            return None

        status_name = trading_pb2.OrderStatus.Name(response.status).removeprefix("ORDER_STATUS_")
        self.log.log(
            "order_result",
            symbol=symbol,
            status=status_name,
            reject_reason=response.reject_reason,
            exchange_order_id=response.exchange_order_id or None,
        )

        filled_qty = _decimal_or_none(response.filled_quantity.value if response.HasField("filled_quantity") else None)
        self.portfolio.on_order_update(client_order_id, symbol, side, status_name, filled_qty)
        return status_name

    async def _manage_resting_order(self, intent: OrderIntent, order_stub) -> None:
        """The maker/limit order state machine institutional audit Phase
        1.3 calls for: place a post-only limit order -> wait up to
        `execution.limit_order_timeout_secs` for it to fill -> if it
        hasn't (fully) filled, cancel it and, for up to
        `execution.limit_reprice_attempts` more rounds, reprice at the
        then-current best bid/ask and try again -> if quantity still
        remains once attempts are exhausted, either submit a MARKET order
        for the remainder (`execution.fallback_to_market=True`) or give up
        on it entirely. A risk/exchange REJECTED response ends the loop
        immediately without retrying — a rejection reason (e.g. a
        guardrail tripping) is very unlikely to have changed by the next
        attempt, so retrying blindly would just be noise.

        Runs as a background task (see _submit_order) so it never blocks
        market-data processing for other symbols while it waits.
        """
        remaining = intent.quantity
        limit_price = intent.limit_price
        max_attempts = self.config.execution.limit_reprice_attempts
        timeout = self.config.execution.limit_order_timeout_secs

        for attempt in range(max_attempts + 1):
            if limit_price is None:
                # No book to price against (shouldn't happen — policy.py
                # only builds a LIMIT intent when it just computed a price
                # off live features — but fail safe rather than submit a
                # priceless "limit" order).
                self.log.log(
                    "resting_order_no_price",
                    symbol=intent.symbol,
                    note="no limit price available to (re)price this attempt — abandoning the maker path",
                )
                break

            client_order_id = f"{self.strategy.strategy_id}-{intent.symbol}-{time.time_ns()}"
            event = asyncio.Event()
            self._resting_order_events[client_order_id] = event
            self._pending_orders[client_order_id] = (intent.symbol, intent.side)

            order = trading_pb2.OrderRequest(
                client_order_id=client_order_id,
                symbol=intent.symbol,
                exchange=self.config.strategy.exchange,
                side=_side_to_pb(intent.side),
                type=trading_pb2.ORDER_TYPE_LIMIT,
                quantity=trading_pb2.Decimal(value=str(remaining)),
                limit_price=trading_pb2.Decimal(value=str(limit_price)),
                strategy_id=self.strategy.strategy_id,
                post_only=True,
            )
            self.log.log(
                "resting_order_submitted",
                symbol=intent.symbol,
                side=intent.side,
                quantity=str(remaining),
                limit_price=str(limit_price),
                attempt=attempt,
                signal=round(intent.signal, 4),
            )
            try:
                response = await order_stub.SubmitOrder(order)
            except grpc.aio.AioRpcError as e:
                self.log.log("resting_order_submit_error", symbol=intent.symbol, detail=str(e), attempt=attempt)
                self._cleanup_resting_order(client_order_id)
                return

            status_name = trading_pb2.OrderStatus.Name(response.status).removeprefix("ORDER_STATUS_")
            if status_name == "REJECTED":
                self.log.log(
                    "resting_order_rejected",
                    symbol=intent.symbol,
                    reason=response.reject_reason,
                    attempt=attempt,
                )
                self._cleanup_resting_order(client_order_id)
                return

            # Wait for a fill/cancel update to arrive on the stream, up to
            # the configured timeout. A timeout here is the expected,
            # common case (the order just hasn't filled yet), not an
            # error.
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass

            update = self._resting_order_last_update.get(client_order_id)
            if update is not None and update.HasField("filled_quantity"):
                filled = _decimal_or_none(update.filled_quantity.value)
                if filled:
                    remaining -= filled
            self._cleanup_resting_order(client_order_id)

            if remaining <= 0:
                self.log.log("resting_order_filled", symbol=intent.symbol, quantity=str(intent.quantity))
                return

            # Not (fully) filled — cancel whatever's left resting before
            # repricing or giving up. A no-op if it already
            # filled/canceled/was rejected in the moment between the wait
            # above timing out and this call landing (CancelOrder reports
            # ALREADY_CLOSED rather than erroring — see trading.proto).
            try:
                cancel_response = await order_stub.CancelOrder(
                    trading_pb2.CancelOrderRequest(client_order_id=client_order_id)
                )
                self.log.log(
                    "resting_order_canceled",
                    symbol=intent.symbol,
                    outcome=trading_pb2.CancelOutcome.Name(cancel_response.outcome),
                    remaining=str(remaining),
                    attempt=attempt,
                )
            except grpc.aio.AioRpcError as e:
                self.log.log("resting_order_cancel_error", symbol=intent.symbol, detail=str(e), attempt=attempt)

            if attempt >= max_attempts:
                break

            # Reprice at the (likely moved) current best bid/ask ahead of
            # the next attempt — the price computed at decision time is
            # stale by now.
            latest = self._latest_book.get(intent.symbol)
            limit_price = (latest[0] if intent.side == "BUY" else latest[1]) if latest is not None else None

        if remaining <= 0:
            return

        if self.config.execution.fallback_to_market:
            self.log.log(
                "resting_order_fallback_to_market",
                symbol=intent.symbol,
                quantity=str(remaining),
                note="exhausted reprice attempts still unfilled — falling back to a MARKET order per "
                "execution.fallback_to_market=true",
            )
            await self._submit_market_order(intent.symbol, intent.side, remaining, intent.signal, order_stub)
        else:
            self.log.log(
                "resting_order_abandoned",
                symbol=intent.symbol,
                quantity=str(remaining),
                note="exhausted reprice attempts, execution.fallback_to_market=false — remaining quantity "
                "was never filled",
            )

    def _cleanup_resting_order(self, client_order_id: str) -> None:
        self._resting_order_events.pop(client_order_id, None)
        self._resting_order_last_update.pop(client_order_id, None)
        self._pending_orders.pop(client_order_id, None)

    async def _order_update_loop(self, order_stub) -> None:
        # Real fills/cancellations arrive here from Kraken's private feed
        # (see rust-core/src/kraken_private_ws.rs and order.rs's
        # StreamOrderUpdates) — this is not a stub. Every update this
        # process receives is attributed via `_pending_orders`, keyed by
        # the client_order_id assigned at submission time.
        #
        # An order actively managed by `_manage_resting_order`
        # (institutional audit Phase 1.3) is looked up rather than popped,
        # since it may legitimately receive more than one update (a
        # partial fill followed later by another partial or a final fill/
        # cancel) — `_manage_resting_order` itself is responsible for
        # cleaning up via `_cleanup_resting_order` once it's done with that
        # order. A plain (non-resting) order is still popped on its first
        # update, same as before — it should only ever get exactly one.
        request = trading_pb2.StreamOrderUpdatesRequest(strategy_id=self.strategy.strategy_id)
        try:
            async for update in order_stub.StreamOrderUpdates(request):
                is_managed = update.client_order_id in self._resting_order_events
                pending = (
                    self._pending_orders.get(update.client_order_id)
                    if is_managed
                    else self._pending_orders.pop(update.client_order_id, None)
                )
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
                self.portfolio.on_order_update(update.client_order_id, symbol, side, status_name, filled_qty)

                if is_managed:
                    self._resting_order_last_update[update.client_order_id] = update
                    self._resting_order_events[update.client_order_id].set()
        except grpc.aio.AioRpcError as e:
            self.log.log("order_update_stream_error", detail=str(e))


def _decimal_or_none(value: Optional[str]) -> Optional[Decimal]:
    return Decimal(value) if value else None


def _side_to_pb(side: Side):
    return trading_pb2.ORDER_SIDE_BUY if side == "BUY" else trading_pb2.ORDER_SIDE_SELL


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
