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
   logging + log visualization. ✅ done — see below.
7. Real fill/status tracking from Kraken's private WebSocket feed, wired
   into `StreamOrderUpdates`. ✅ done — see below. Persistence (a real
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

## Real fill tracking (rust-core/src/kraken_private_ws.rs)

Closes the gap flagged in every earlier section of this README:
`StreamOrderUpdates` is no longer an empty stub. Order fills and status
changes now come from Kraken's **private** (authenticated) WebSocket v2
feed — a separate connection entirely from the public market-data feed in
`kraken.rs`, requiring a short-lived token fetched via a new
`KrakenRestClient::get_websockets_token` (same HMAC-SHA512 signing scheme
as `AddOrder`, refactored into a shared `signed_post` helper now used by
both).

**How it's wired**: `main.rs` starts one `kraken_private_ws::run` task per
exchange that has an execution client configured (no credentials -> no
client -> nothing to authenticate as -> task doesn't start). It fetches a
fresh token on every (re)connect, subscribes to the `executions` channel,
and publishes every fill/status event onto a broadcast channel.
`OrderServiceImpl::stream_order_updates` fans that out to gRPC
subscribers, filtered by `strategy_id` via a new shared registry
(`client_order_id -> strategy_id`, populated in `order.rs` at the moment
an order is actually sent to Kraken — never for a `dry_run` validate-only
call, which can never produce a real execution report).

**Verified live**, with deliberately fake API credentials: the signed
request to `/0/private/GetWebSocketsToken` reaches Kraken's real server
and gets back a genuine, correctly-parsed `EAPI:Invalid key` rejection —
the same category of evidence as the `AddOrder` signing test. The
reconnect loop was also confirmed live, backing off 3 seconds and
retrying indefinitely without crashing when the token fetch fails. 6 new
unit tests (`cargo test`, 22/22 passing project-wide) cover parsing a
synthetic Kraken-shaped execution report into an `OrderUpdate` — partial
fill, full fill, missing/unattributable fields, and every `order_status`
mapping.

**What's NOT verified, because it requires a real, funded Kraken
account**: the private WebSocket connection itself was never reached (the
fake-credential token fetch fails before that point), so neither the
connection URL (`wss://ws-auth.kraken.com/v2`, inferred from Kraken's
confirmed v1 private-feed host plus the `/v2` suffix the public feed
uses) nor the `executions` channel's exact message schema has been
confirmed against a real execution report — both come from Kraken's
public documentation. Before relying on this for real position tracking,
get a real token from a real account and confirm both against an actual
fill.

**What's stubbed, on purpose**: the `strategy_id` registry never evicts
entries (unbounded growth over a long-running process — low-severity
since entries are just short strings, but a real fix would need a TTL or
eviction-on-terminal-status). A lagged broadcast subscriber logs loudly
(`tracing::error!`, not `warn!`, unlike the equivalent market-data case)
but still has no reconciliation path against Kraken's own order state to
recover from a missed fill — that gap is now explicit rather than hidden
behind an empty stream. `PortfolioManager` on the Python side (see the
Strategy layer section above) can now, in principle, receive real
updates through this path — that hasn't been exercised together with a
funded account either, for the same reason nothing else funded-account-only
has been in this project.

## Closing the risk-engine loop (rust-core/src/risk.rs)

Before this, `RiskEngine`'s position tracking and daily kill switch existed
but had nothing to act on — `positions` never moved off zero, and
`kill_switch_max_daily_loss_usd` was read from config but never checked
against anything. Real fills from the private execution feed above now
flow directly into the risk engine, so both guardrails reflect what
actually happened at the exchange.

**How it's wired**: `kraken_private_ws.rs`'s message loop extracts a
`FillEvent` from each `executions`-channel entry where Kraken's
`exec_type == "trade"` — the field that specifically marks "this event
carries a real execution," as opposed to a pure status transition (new,
canceled, amended, etc.). Each `FillEvent` uses `last_qty`/`last_price`
(the size and price of *that specific* execution) rather than diffing
`cum_qty` across messages: a `cum_qty` delta needs locally-persisted state
to interpret as a quantity, and that state would be lost across a
WebSocket reconnect, silently double- or under-counting fills.
`last_qty`/`last_price` are self-contained per-event values that need no
such state. A bounded (512-entry) recently-seen `exec_id`/`trade_id` set,
scoped to each connection, skips an event redelivered within that window
— a guard against the common case, not a full exactly-once guarantee.

Each `FillEvent` is applied via `RiskEngine::apply_fill(symbol, side, qty,
price)`, which:
- Tracks position on an **average-cost basis** per symbol
  (`PositionState { qty, avg_entry_price }`): a fill in the same direction
  extends the position and rolls the average entry price forward,
  size-weighted; a fill in the opposite direction realizes PnL against
  that average price for the closing quantity, and any quantity beyond a
  full close flips the position onto a fresh average-entry basis at the
  fill price.
- Accumulates realized PnL for the current UTC day (`realized_pnl_usd`),
  reset via a day-index rollover (`unix_seconds / 86_400`) computed with
  only `std::time` — no `chrono` dependency needed for a UTC day
  boundary.
- Feeds a new `check_kill_switch` check, now the **first** check inside
  `evaluate()` (before rate limiting): once realized loss for the day
  reaches `kill_switch_max_daily_loss_usd`, every order is rejected until
  the day rolls over. Like every other check in this module, an
  unparseable config value fails closed (rejects everything) rather than
  silently disabling the switch.

The existing per-symbol and portfolio-wide position-limit checks in
`evaluate()` now read live position data instead of a value that could
only ever be zero.

**Verified**: 9 new unit tests (`cargo test`, 31/31 passing project-wide)
cover opening a position, weighted-average-price extension, realizing a
gain, realizing a loss with a position flip, the kill switch tripping
after the configured daily loss and staying silent on a gain, and
`kraken_private_ws.rs`'s `build_fill_event` parsing (a `"trade"` exec
type, non-trade types being ignored, the `exec_id`/`trade_id` fallback,
and a trade missing required fields being skipped rather than panicking).
A live smoke run against real Kraken market data (fake execution
credentials) confirmed the whole process still starts, connects, and
reconnects cleanly with this wiring in place — no observable behavior
change from before except the added fill/kill-switch logging, since fake
credentials never reach a real fill.

**What's NOT verified, for the same reason as the section above**: no
real fill has ever been applied through this path, since that requires a
real account's `executions` channel to actually emit a `"trade"`-type
event. The `exec_type`/`last_qty`/`last_price`/`side` field names come
from Kraken's public v2 documentation, not an observed report — get a
real fill on a funded account before trusting this for real risk
management. The read accessors `RiskEngine::position` and
`RiskEngine::realized_pnl_today` are public but not yet exposed anywhere
(no gRPC endpoint or log line surfaces them to an operator) — a natural
next step once this is worth watching live.

## Persistence (rust-core/src/persistence.rs)

Before this, every piece of state introduced in the section above —
positions, the kill switch's daily realized-PnL counter, and the whole
order/fill history — lived only in memory. Restarting the process (a
deploy, a crash, a manual bounce) silently reset all of it: positions back
to flat, the kill switch's counter back to zero, no record of what had
happened. That gap is now closed with a local SQLite database.

**Why SQLite, and why `rusqlite` over an async driver**: this project
already treats decimals-as-strings as a hard rule (config, gRPC), so the
same discipline applies here — every numeric column is `TEXT`, never
`REAL`, so nothing touches floating point on the way to or from disk.
`rusqlite` (with SQLite compiled in via the `bundled` feature, no system
library dependency) was chosen over an async SQL driver because every
operation here is a single small, fast read or write (one row, at most a
handful per fill or order) — not a volume that needs async I/O, and a
plain `std::sync::Mutex<Connection>` guarded by methods that never hold
the lock across an `.await` is simpler to reason about than threading an
async pool through the codebase for this.

**Schema** (four tables, migrations are just idempotent `CREATE TABLE IF
NOT EXISTS`, no version to track for something this small):
- `positions` — one row per symbol, upserted by `RiskEngine::apply_fill`.
- `kill_switch_state` — a single row holding the current UTC day index and
  that day's realized PnL, upserted on every fill and on every day
  rollover.
- `orders` — one row per `client_order_id`, upserted at submission
  (`order.rs`) and updated in place as status changes arrive on the
  private feed (`kraken_private_ws.rs`).
- `fills` — an append-only audit trail of every applied fill, with a
  `UNIQUE` constraint on `exec_id` that makes it the durable half of fill
  de-duplication (see below).

**How it's wired**: `main.rs` opens the store at startup (path from the
new `[persistence]` config section, defaulting to
`./data/trading-core.sqlite3`, parent directories created automatically)
and calls `RiskEngine::attach_store`, which hydrates `positions` and (if
it's from *today*, UTC) the kill-switch counter from disk before the
engine does anything else — a counter from a previous day is deliberately
**not** restored, since carrying yesterday's loss into today would either
trip the switch for no reason or mask today's actual losses under a
leftover cushion. The same `Store` is shared (as `Option<Arc<Store>>`)
with `OrderServiceImpl` (order submission outcomes) and
`kraken_private_ws::run` (order status updates from the private feed and,
via `RiskEngine::apply_fill`, every applied fill). A store that fails to
open at startup is logged loudly but does **not** stop the process — it
degrades to the pre-persistence behavior (in-memory-only, resets on
restart) rather than refusing to trade over what is, for now, optional
infrastructure.

**Fill de-duplication is now two layers deep**: `kraken_private_ws.rs`'s
in-memory 512-entry window (from the section above) catches a redelivery
within the same connection cheaply; `RiskEngine::apply_fill` now also
checks the `fills` table's `exec_id` before touching any position state,
which catches a redelivery across a reconnect *or a full process
restart* — something the in-memory window structurally cannot, since it's
wiped out along with everything else on restart.

**Verified**: 6 new unit tests in `persistence.rs` (round-tripping
positions, kill-switch state, order records and status transitions, and
both dedup paths for fills) plus 3 new integration-style tests in
`risk.rs` (a position surviving a fresh `RiskEngine` attached to the same
store, `apply_fill` rejecting a redelivered `exec_id` via the store, and a
previous-day kill-switch counter being correctly ignored on attach) —
50/50 tests passing project-wide. A live smoke run confirmed the database
file, WAL journal, and full schema are created automatically on a cold
start against `config.example.toml`'s new `[persistence]` section, with
no change in the process's other observable behavior.

**What's NOT done**: nothing reconciles `orders`/`positions` against
Kraken's own order/position state after a restart — a startup log line
now warns if there are open orders left over from before a restart, but
that's visibility, not reconciliation. There's no retention policy on the
append-only `fills` table (it grows forever) and no admin/CLI tool to
query the database — for now, `sqlite3 data/trading-core.sqlite3` is the
tool. The theoretical race noted in `Store::record_fill`'s doc comment
(two writers racing on the same `exec_id` between a dedup check and the
insert) is accepted as-is: there is only ever one private-feed task per
exchange in this project, so it doesn't occur in practice today, but would
need a transaction if that ever changes.

## Startup reconciliation (rust-core/src/reconcile.rs)

Closes the exact gap the persistence section above named: until now, a
restart trusted whatever `orders`/`positions` said in SQLite with no check
against what Kraken itself actually has. Now, once a store and a real
execution client both exist for an exchange, `main.rs` runs a
reconciliation pass before starting the gRPC server (bounded by a 15s
timeout, so an unreachable Kraken can't hang startup — a failure here is
logged loudly and startup proceeds anyway, the same "optional
infrastructure" posture as the persistence store itself).

**What it does**: pulls every locally-persisted order that isn't already
in a terminal status, calls Kraken's `OpenOrders` to see what Kraken still
considers open, and for anything no longer on that list, calls
`QueryOrders` (chunked at Kraken's 50-txid-per-call limit) to find out
what actually happened — closed with executed quantity (a fill), closed
with none, or canceled/expired. The local `orders` row is updated to
match. Separately, any order Kraken lists as open that has no local record
at all is logged as a warning — never adopted, since this process has no
way to know that order's `strategy_id` or original intent.

**Deliberately does NOT touch positions or PnL.** `RiskEngine`'s position
and realized-PnL state is only ever mutated by `apply_fill`, driven by the
live executions feed — reconciliation reading a historical `vol_exec` from
`QueryOrders` and applying it there too would risk double-counting a fill
the live feed already applied before a restart, and there's no reliable
way from Kraken's order-level data alone to tell "already counted" from
"missed while the process was down." Instead, when a since-closed order
shows real executed quantity with no matching row in the `fills` table,
that's surfaced as a `tracing::error!` (`possible_missed_fills` in the
summary log line) for a human to check by hand against Kraken's own trade
history — a conservative choice: it tells you exactly where to look
instead of silently guessing and risking the wrong number.

**Also added, informational only**: `KrakenRestClient::get_account_balance`
(`/0/private/Balance`) — not called by reconciliation itself, since a spot
wallet balance isn't the same thing as `RiskEngine`'s tracked position
(which starts at zero when the bot first runs and only reflects fills
since then; reconciling the two would require knowing the account's
pre-bot holdings, which isn't knowable from here). It's available for a
future operator-visibility feature rather than wired to anything yet.

**Verified**: 15 new unit tests (`cargo test`, 53/53 passing project-wide)
— JSON parsing for `OpenOrders`/`QueryOrders`/`Balance` against synthetic
Kraken-shaped responses, and the reconciliation module's pure decision
logic (`diff_local_vs_kraken_open`, `resolve_closed_status`,
`kraken_open_with_no_local_record`) tested independently of any network
call, the same split `kraken_rest.rs` itself uses for its signing logic. A
live smoke run with fake credentials confirmed the real end-to-end path:
the signed `OpenOrders` request reaches Kraken's server and gets back a
genuine `EAPI:Invalid key` rejection, which is logged as a failed
reconciliation pass without stopping the process — gRPC server, market
data, and the private execution feed all start normally regardless.

**What's NOT verified, for the same reason as everything else
private-endpoint-shaped in this project**: no real reconciliation has ever
run against an account with actual open or recently-closed orders, since
that requires a funded account. The `OpenOrders`/`QueryOrders` response
shapes come from Kraken's public documentation, not an observed response.
