"""
Exercises OrderService.SubmitOrder against the live Rust server to prove
the risk engine actually runs over gRPC, not just in Rust unit tests.

Run:
    python -m strategy.order_client
"""

import logging

import grpc

from strategy.pb import trading_pb2, trading_pb2_grpc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("order_client")

RUST_CORE_ADDR = "localhost:50051"


def submit(stub, label: str, symbol: str = "BTC-USD", **kwargs) -> None:
    order = trading_pb2.OrderRequest(
        client_order_id=f"test-{label}",
        symbol=symbol,
        exchange="kraken",
        side=trading_pb2.ORDER_SIDE_BUY,
        strategy_id="manual-test",
        **kwargs,
    )
    log.info("--- %s ---", label)
    try:
        response = stub.SubmitOrder(order)
        log.info(
            "response: status=%s reject_reason=%r",
            trading_pb2.OrderStatus.Name(response.status),
            response.reject_reason,
        )
    except grpc.RpcError as e:
        log.info("gRPC error: %s — %s", e.code(), e.details())


def run() -> None:
    with grpc.insecure_channel(RUST_CORE_ADDR) as channel:
        stub = trading_pb2_grpc.OrderServiceStub(channel)

        # Comfortably within every configured limit for BTC-USD.
        submit(
            stub,
            "within_limits",
            type=trading_pb2.ORDER_TYPE_LIMIT,
            quantity=trading_pb2.Decimal(value="0.001"),
            limit_price=trading_pb2.Decimal(value="30000"),
        )

        # max_order_size for BTC-USD is 0.05 — this should be rejected.
        submit(
            stub,
            "over_size_limit",
            type=trading_pb2.ORDER_TYPE_LIMIT,
            quantity=trading_pb2.Decimal(value="0.5"),
            limit_price=trading_pb2.Decimal(value="30000"),
        )

        # max_order_notional_usd for BTC-USD is 2000 — 0.04 * 60000 = 2400.
        submit(
            stub,
            "over_notional_limit",
            type=trading_pb2.ORDER_TYPE_LIMIT,
            quantity=trading_pb2.Decimal(value="0.04"),
            limit_price=trading_pb2.Decimal(value="60000"),
        )

        # A symbol that isn't in config.example.toml at all.
        submit(
            stub,
            "unconfigured_symbol",
            symbol="DOGE-USD",
            type=trading_pb2.ORDER_TYPE_LIMIT,
            quantity=trading_pb2.Decimal(value="1"),
            limit_price=trading_pb2.Decimal(value="0.1"),
        )


if __name__ == "__main__":
    run()
