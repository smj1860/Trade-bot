"""
Minimal proof-of-pipe client.

Connects to the Rust trading-core gRPC server and subscribes to market data.
Right now the Rust side streams nothing (the order book engine isn't wired
up yet), so this will just sit connected and log that the subscription was
accepted — that's enough to prove the IPC layer works end-to-end before any
strategy or model logic goes in.

Run:
    python -m strategy.client
"""

import logging

import grpc

from strategy.pb import trading_pb2, trading_pb2_grpc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("client")

RUST_CORE_ADDR = "localhost:50051"


def run() -> None:
    with grpc.insecure_channel(RUST_CORE_ADDR) as channel:
        stub = trading_pb2_grpc.MarketDataServiceStub(channel)

        # Empty symbols list = "everything this Rust instance tracks", per
        # the contract in trading.proto.
        request = trading_pb2.SubscribeRequest(symbols=[])
        log.info("subscribing to market data on %s", RUST_CORE_ADDR)

        try:
            for event in stub.SubscribeMarketData(request):
                if event.HasField("order_book_update"):
                    u = event.order_book_update
                    log.info(
                        "order book update: %s best_bid=%s best_ask=%s",
                        u.symbol,
                        u.bids[0].price.value if u.bids else None,
                        u.asks[0].price.value if u.asks else None,
                    )
                elif event.HasField("metric"):
                    m = event.metric
                    log.info("metric: %s.%s = %s", m.symbol, m.name, m.value.value)
        except grpc.RpcError as e:
            log.error("stream ended: %s", e)


if __name__ == "__main__":
    run()
