# Crypto Trading Bot — Scaffold

## Layout

```
proto/
  trading.proto          gRPC contract between the Rust and Python processes
config/
  config.example.toml    per-symbol, currency-agnostic configuration
rust-core/                Process A: WebSocket ingest, order book, risk, execution
python-strategy/           Process B: strategy, ML, portfolio management
```

## Design decisions locked in

- **Symbol format:** normalized as `BASE-QUOTE` (e.g. `BTC-USD`) everywhere
  internally. Each exchange adapter translates its own native format to/from
  this at the boundary — nothing upstream of the adapter ever sees
  exchange-native symbol strings.
- **Currency-agnostic:** no asset is hardcoded anywhere. Symbols, tick/lot
  size, and per-symbol risk limits are all config-driven (`config.toml`).
  Adding a new coin is a config change, not a code change.
- **Precision:** prices/quantities are `Decimal` (string-encoded over gRPC,
  `rust_decimal::Decimal` in Rust, `decimal.Decimal` in Python) — never
  floats, to avoid rounding error compounding across order book deltas.
- **Risk guardrails live in Rust**, not Python. Python can request an order;
  Rust is the last line of defense that can reject it (position limits,
  order size limits, per-symbol and global caps, a daily-loss kill switch).
- **gRPC over shared memory** for the IPC layer — much faster to build and
  debug, and not a proven bottleneck yet. Revisit only if profiling shows
  gRPC overhead is actually the limiting factor.

## Next steps

1. Stand up the Rust gRPC server skeleton (tonic) that compiles the
   `.proto` and exposes empty `MarketDataService` / `OrderService` impls.
   ✅ done — builds and runs.
2. Wire one exchange WebSocket into an order-book struct, streamed out over
   `SubscribeMarketData`. Exchange: **Kraken** (WebSocket API v2,
   `wss://ws.kraken.com/v2`, pairs like `BTC/USD`). ✅ done — connects,
   subscribes to the `book` channel, maintains order book state, and was
   verified live: real BTC-USD/ETH-USD order book updates streamed
   through Rust → gRPC → Python during testing.
3. Build a minimal Python client that subscribes and just logs — proves the
   pipe end-to-end before any strategy logic goes in. ✅ done — tested live
   against the running Rust server, receiving real market data.
4. Risk guardrail enforcement in Rust. ✅ done — see below.
5. Kraken REST order execution client — wires an Approved order to
   Kraken's real `/0/private/AddOrder` endpoint. ✅ done — see below.
6. Strategy/model layer, client-side portfolio tracking, structured
   logging + log visualization. ✅ done — see below. Persistence (a real
   database instead of in-memory state) still not started.

## Notes from building this

- **Order book state** (`rust-core/src/orderbook.rs`) is unit-tested —
  snapshot replace, delta update, zero-quantity level removal, and a
  bid/ask-never-cross sanity check. Run with `cargo test`.
- **Reconnect logic**: the Kraken adapter loops forever, reconnecting with
  a 3s delay on any error (parse failure, dropped connection, etc.) rather
  than taking the whole service down. Verified by forcing a bad TLS trust
  chain during testing — it logged and retried cleanly instead of
  crashing.
- **TLS trust store**: uses native OS trust roots (`rustls-tls-native-roots`)
  rather than a bundled public-CA list. This is the right choice for a
  service that might run behind a corporate/cloud egress proxy — it trusts
  whatever CAs the host OS trusts, same as `curl` would, rather than a
  fixed list that needs a rebuild to update.
- **Precision**: Kraken sends prices/quantities as JSON numbers. To avoid
  losing precision by round-tripping through `f64`, `serde_json`'s
  `arbitrary_precision` feature is enabled so the exact original digits
  are parsed straight into `Decimal`.
- **Not yet wired**: Kraken's per-message checksum (for detecting a
  desynced book) and its timestamp field are parsed by nothing yet — the
  proto fields exist (`sequence`, `exchange_timestamp_ns`) but are sent as
  0/placeholder. Worth doing before this goes anywhere near real capital.

## Risk guardrails (rust-core/src/risk.rs)

`OrderService.SubmitOrder` now runs every incoming order through a real
risk engine before anything else happens. Enforced, in this order:

1. Global order rate limit (orders/minute, config-driven)
2. Symbol must be configured and enabled for the requesting exchange
3. Order size ≤ per-symbol `max_order_size`
4. Order notional (qty × price) ≤ per-symbol `max_order_notional_usd`
5. Projected position after this order ≤ per-symbol `max_position_usd`
6. Projected combined portfolio exposure ≤ global `max_total_position_usd`

For a market order (no limit price), the price used for #4–6 comes from
the **live order book** — the same shared state the market data stream
reads from, not a stale or hardcoded number. Tested live: with no book
populated yet, a market order is correctly rejected ("no price
available"); once Kraken data is flowing, it prices against the real
best bid/ask.

A rejection is a normal, successful gRPC response (`OrderStatus.REJECTED`
with a `reject_reason` string) — rejecting bad orders is the risk
engine doing its job, not an infrastructure failure. An order that
**passes** every check gets `Status::unimplemented` instead of a fake
acceptance, because there is still no exchange execution client — nothing
here will ever claim an order reached Kraken when it didn't.

**What's stubbed, on purpose**: per-symbol positions start at zero and
never move, because nothing updates them without real fills (no execution
client yet). The position-limit and portfolio-exposure checks are real
and tested — they just have nothing but zero to work from until
execution exists. The daily-loss kill switch
(`kill_switch_max_daily_loss_usd`) is parsed from config but not
enforced yet, for the same reason: no PnL exists to check it against.

7 unit tests cover the risk engine (`cargo test`) — symbol validation,
size/notional/position limits, market-order pricing off the live book,
and rate limiting. `python-strategy/strategy/order_client.py` exercises
the same logic over real gRPC calls against the running server.

## Execution client (rust-core/src/kraken_rest.rs)

An order that passes every risk check now actually gets sent to Kraken,
via `POST /0/private/AddOrder`, signed with Kraken's documented
HMAC-SHA512 scheme (`HMAC(base64_decode(api_secret), path + SHA256(nonce
+ post_data))`, base64-encoded, sent as the `API-Sign` header).

**Safe by default.** `[execution] dry_run` in config controls Kraken's
own `validate` flag on every request, and it defaults to `true` even if
the whole `[execution]` section is deleted from the config file —
a config that says nothing about execution can never be silently read
as permission to trade real money. With `validate=true`, Kraken checks
and responds to the request exactly as it would for real, but never
places it on the order book.

**No credentials, no client, no crash, no fake success.** `main.rs` only
builds a `KrakenRestClient` for an exchange when `KRAKEN_API_KEY` /
`KRAKEN_API_SECRET` are both set as environment variables (never in
config files, so they can't end up committed). If they're missing, an
order that passes risk checks comes back as a gRPC `FAILED_PRECONDITION`
— "no execution client is configured" — rather than being silently
dropped or reported as submitted.

**What was actually tested, live, against Kraken's real server**: the
full request/response pipeline. With deliberately fake API credentials
and `dry_run=true`, a risk-approved BTC-USD order was submitted through
Python → gRPC → the risk engine → the Kraken REST client → a real HTTPS
POST to `api.kraken.com` → Kraken's genuine JSON error response
(`EAPI:Invalid key`) → parsed and forwarded back through gRPC as an
`ORDER_STATUS_REJECTED` with that exact reason. That confirms the HTTP
request is built correctly, actually reaches Kraken, and Kraken's
response is parsed and surfaced correctly. It does **not** confirm the
HMAC signature is byte-correct against a real, authenticated Kraken
account — that requires real API credentials, which don't exist in this
sandbox. The 5 unit tests on the signing function (`cargo test`) only
check it's deterministic and sensitive to its inputs (nonce, POST body,
URI path) — not that it matches Kraken's server-side computation.

**Known unverified detail — REST pair naming.** Kraken's WebSocket v2
API and its REST AddOrder endpoint use different pair-naming schemes
(`"BTC/USD"` for WS v2 vs. an altname like `"XBTUSD"` for REST). A new
`rest_native_symbol` config field carries the REST name separately from
`exchange_native_symbol` (the WS name), but the actual altnames in
`config.example.toml` are my best understanding from Kraken's public
docs, not values I could confirm against a live account — verify them
against Kraken's `/0/public/AssetPairs` endpoint before trading live.

**What's stubbed, on purpose**: fills aren't tracked yet (no order
lifecycle streaming from Kraken back into `positions` in the risk
engine), so the position-limit checks still start every symbol at zero
regardless of live orders placed. `StreamOrderUpdates` is still an empty
stream. The daily-loss kill switch is still unenforced. All of this was
true before this step and remains true after it — this step only wired
up the "send an approved order to the exchange" half, not the "track
what happened to it afterward" half.

## Strategy layer (python-strategy/strategy/)

Process B, end to end: consumes `SubscribeMarketData`, computes features,
runs them through a model, turns a signal into an order (or a no-op),
tracks position client-side, and logs every step. Config lives in
`strategy_config.example.toml` (copy to `strategy_config.toml`), separate
from Rust's `config/config.example.toml`.

**Pipeline, one module per stage:**
- `features.py` — turns raw order-book updates into a small feature
  vector per symbol: mid-price, spread, order-book imbalance (top-of-book
  only, not full depth), and momentum (mid-price now vs. N updates ago).
- `models.py` — `ModelWrapper` interface (`predict(features) -> signal in
  [-1, 1]`). `RuleBasedModel` is the only one actually driving decisions
  right now: a fixed weighted sum of imbalance and momentum, every
  coefficient visible in config. `SklearnModelWrapper` and
  `TorchModelWrapper` are real, tested implementations of the same
  interface — tested against a toy classifier trained on nothing
  meaningful (see `tests/test_models.py`), not validated as trading
  models, because there's no trained model or historical data in this
  project yet. Swapping to a real model later is a `strategy_config.toml`
  change (`model.kind`, `model.model_path`, `model.feature_order`), not a
  strategy rewrite.
- `policy.py` — turns a signal into an `OrderIntent` (or nothing): a
  signal threshold, a per-symbol cooldown, and a soft client-side position
  limit that's advisory only — Rust's risk engine is still the real
  enforcement point, never duplicated here.
- `portfolio.py` — tracks net position per symbol from confirmed fills
  only (`FILLED`/`PARTIALLY_FILLED`), never from `ACCEPTED`. Honest
  limitation: Rust's `StreamOrderUpdates` is still an empty-stream stub,
  so there's no real fill-notification pipeline yet — this is built and
  unit-tested against the data shape it'll need, not validated against
  live fills, because there aren't any yet.
- `imbalance_momentum.py` — the first concrete `Strategy`, wiring the
  above together. Genuinely runnable against live Kraken data today.
- `engine.py` — the async process: `grpc.aio` streams from both Rust
  services concurrently, feeds order-book updates to the strategy,
  submits any resulting order, and logs everything via
  `logging_utils.py` to `logs/strategy.jsonl` (one JSON object per event:
  `signal`, `order_would_submit`/`order_submitted`, `order_result`, etc).

**Safety, layered on top of Rust's own dry-run flag**: `execution.dry_run_only`
in `strategy_config.toml` (default `true`) stops the engine from ever
calling `SubmitOrder` at all — it computes and logs exactly what it would
have sent (`order_would_submit`) and stops there. Nothing reaches the
network. Rust's own `execution.dry_run` is a second, independent gate
further downstream.

**Verified live**, running the engine in paper mode against the real Rust
server and real Kraken market data: real order-book updates flowed
through the whole pipeline into features, a signal, and (past the
threshold) an `OrderIntent`. The per-symbol cooldown was confirmed
working under real load — roughly 850 qualifying ticks arrived over ~30
seconds and exactly one order-intent fired per symbol per cooldown
window, not 850. `scripts/plot_log.py` was run against that real log and
correctly rendered mid-price with buy/sell markers per symbol.

**What's NOT verified**: no real order was ever submitted from the
strategy layer end-to-end in this test (paper mode, by design) — the
execution path itself (Python → Rust → Kraken) was already verified
separately in the execution-client step above, using `order_client.py`
directly. The `imbalance_momentum` rule is not a validated trading
strategy in any predictive sense — it wasn't backtested, and there's no
historical data in this project to backtest it against. Treat it as
working plumbing with an honest, simple rule attached, not as a strategy
with an edge.

**Testing**: 34 unit tests (`pytest python-strategy/tests/`) cover
features, models (including the sklearn wrapper against a toy model),
policy (threshold, cooldown, soft position limits), and portfolio
tracking — all pure logic, no gRPC or live server required.

**Dependencies**: `requirements.txt` (core — just `grpcio`/`grpcio-tools`,
already needed for the existing gRPC clients) stays deliberately light;
`requirements-ml.txt` (scikit-learn, joblib, torch) is only needed if
`strategy.model.kind` is set to `"sklearn"` or `"torch"`; `requirements-dev.txt`
(pytest, matplotlib) is only needed to run tests or `scripts/plot_log.py`.

**What's stubbed, on purpose**: no persistence — positions and the
cooldown/order-history state live in memory and reset on restart. No
backtesting harness. No real ML model. Order-book imbalance uses
top-of-book only, not full depth. All reasonable next steps, none of
them done here.
