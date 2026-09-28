# Execution guardrails: slippage/spread protection, a volatility circuit breaker, and REST rate limiting

Added 2026-09-27, per Stephen's explicit proposal: "Implement Dynamic
Volatility-Based Circuit Breakers & Pre-Trade Rate/Slippage Guardrails in
rust-core." The existing `risk.rs` guardrails (order size, order notional,
per-symbol position cap, portfolio-wide exposure cap, a daily-loss kill
switch, a flat orders-per-minute rate limit) all check *how much* an order
asks for, never *what the market currently looks like*. None of them would
catch a liquidity vacuum that makes a market order's actual fill price far
worse than the top-of-book price, a spread that has blown out well past its
own recent normal, a sudden volatility spike (flash crash or exchange
disruption), or this process itself tripping Kraken's own rate limiter and
getting orders silently dropped.

## What shipped

**New module `rust-core/src/guardrails.rs`** — pure, stateless computation
functions (each with its own unit tests), consuming data the caller already
has rather than tracking anything of their own:

- `simulate_fill_price(levels, qty)` — walks real order-book levels
  (`OrderBook::bid_levels`/`ask_levels`, best price first) consuming `qty`,
  returning the size-weighted average fill price a market order of this
  size would actually get. Returns `None` if the book doesn't have enough
  depth to fill the order at all — treated by the caller as "reject,
  book's too thin to estimate," not "zero cost."
- `slippage_fraction(mid, fill_price)` / `spread_fraction(bid, ask, mid)` —
  small arithmetic helpers.
- `rolling_average_spread(samples, min_samples)` — mean of a book's own
  recent spread history, `None` below `min_samples` (comparing against a
  baseline built from a handful of ticks right after startup would be
  noise, not signal — the caller must read that as "doesn't apply yet,"
  never "baseline is zero").
- `window_parkinson_vol(mid_prices)` — Parkinson's high-low range
  volatility estimate (`sqrt(ln(high/low)^2 / (4 ln 2))`) over a window of
  mid-price ticks — the same formula as `strategy/indicators.py`'s
  `parkinson_vol` on the Python side, but computed from live book ticks
  since this crate has no OHLC bar aggregation of its own.
- `assess_volatility(samples, now, short_window, bucket)` — buckets a
  book's history into `bucket`-sized slices by age, computes each older
  bucket's Parkinson vol as the baseline distribution, and compares the
  most recent `short_window`'s reading against that baseline's mean/stddev
  as a Z-score. Returns `None` below `MIN_BASELINE_BUCKETS` (5) — same
  not-enough-history-yet philosophy as the spread guard.

**`OrderBook` (orderbook.rs) gained a rolling history buffer.** Every
`apply_snapshot`/`apply_update` call that leaves both sides of the book
populated now also records a `PriceSample { at, mid, spread_pct }`,
pruned to the last hour (`HISTORY_MAX_AGE`) with a hard `HISTORY_MAX_SAMPLES`
backstop. This is the one piece of state either new guardrail needs, and
it's fed automatically by ordinary market-data ingestion — no new wiring
into `kraken.rs` or `main.rs` was needed to keep it warm; a guardrail check
at order-evaluation time just reads whatever history has already
accumulated via `OrderBook::recent_samples(max_age)`.

**`risk.rs`'s `evaluate()` gained two new checks**, inserted after the
existing order-size/notional checks and before the position-cap checks:

1. `check_slippage_and_spread` — simulates the fill (walking the side of
   the book the order would actually consume) and rejects if slippage vs.
   mid exceeds `max_slippage_pct`, or if the book doesn't have enough
   depth to fill the order at all; separately rejects if the current
   spread exceeds `spread_multiplier` × its own rolling average (once
   enough history exists).
2. `check_volatility_breaker` — if `assess_volatility` finds the current
   1-minute (configurable) volatility more than `vol_circuit_breaker_stddev`
   standard deviations above its own recent baseline, the symbol freezes
   into **reduce-only/flat** for `vol_circuit_breaker_freeze_secs`: while
   frozen, an order is only approved if it would not increase the
   position's absolute size and would not flip its sign (a flat symbol
   approves nothing at all while frozen — there's nothing to "reduce").
   The freeze deadline is tracked per-symbol in `RiskEngine`
   (`vol_freeze_until`, in-memory only — a freeze is a short-lived
   reaction to a live condition, not state that should survive a restart
   the way positions and the kill switch do).

All five new config fields (`max_slippage_pct`, `spread_multiplier`,
`min_spread_samples`, `vol_circuit_breaker_stddev`, `vol_short_window_secs`,
`vol_baseline_bucket_secs`, `vol_circuit_breaker_freeze_secs`) live under
`[risk.global]` with built-in defaults, so an older `config.toml` that
predates this work keeps loading and working unchanged.

**Kraken REST rate limiting (`kraken_rest.rs`'s new `RateLimiter`)** — an
approximate token-bucket model of Kraken's private-REST call counter: a
counter that grows by `cost_per_call` on each private call, decays
continuously at `decay_per_sec`, and is capped at `max_counter`. A call
that would need to wait longer than `max_wait_secs` to fit fails fast
(`KrakenRestError::RateLimited`) rather than blocking the (time-sensitive)
execution path indefinitely; a shorter wait actually sleeps and retries.
Wired into `KrakenRestClient::signed_post`, so every private call
(`AddOrder`, `OpenOrders`, `QueryOrders`, `Balance`, `GetWebSocketsToken`)
goes through it. Configured via a new `[execution.rate_limit]` section,
with defaults approximating Kraken's documented "Starter" verification
tier (max counter 15, decay ~1/3s) — the most conservative tier, chosen
specifically because this project has never verified which tier a real
account is actually on.

**Honesty about what's approximated, matching this codebase's existing
pattern** (see `kraken_rest.rs`'s top-of-file docs on the signing scheme):
Kraken's real rate-limit model uses a per-endpoint cost table and
tier-dependent caps/decay, none of which has been checked against a real
account here — this implements the *shape* of that model with a single
flat cost, documented as needing re-tuning against Kraken's current docs
(or observed real 429 behavior) before trading live. Likewise, the
volatility breaker's Parkinson estimate is computed from live book
mid-price ticks, not real trade prints (this crate has no OHLC bars of its
own) — a coarser proxy that's fine because it's only ever compared against
its *own* recent history, never an absolute threshold.

## Testing

- `guardrails.rs`: 17 unit tests covering `simulate_fill_price` (blending
  across levels, insufficient depth), slippage/spread fractions, rolling
  average spread (including the not-enough-samples case), Parkinson vol
  (flat/positive/wider-range-reads-higher), and `assess_volatility`
  (flags a genuine spike, small Z-score in a calm market, `None` without
  enough baseline history).
- `kraken_rest.rs`: 4 new `RateLimiter` tests (allows calls under the cap,
  throttles-and-waits within `max_wait`, fails fast when the wait would
  exceed `max_wait`, and the zero-decay edge case).
- `risk.rs`: 9 new integration tests exercising the real `evaluate()` path
  — thin-book slippage rejection, over-threshold slippage rejection,
  approval within tolerance, spread-blowout rejection, the spread guard
  correctly not applying before enough history exists, a genuine
  volatility spike tripping the breaker and freezing new exposure, a
  reduce-only order being approved while frozen (and a flipping order
  still rejected), and no trip in a calm market. The volatility tests
  real-sleep for a few seconds each (`OrderBook`'s history has no
  injectable clock — see `guardrails.rs`'s docs on why) to build
  genuinely time-separated baseline buckets; everything else is
  instant.

228 Rust tests total (up from 60 before this round — most of the increase
is the new modules' own coverage, not regressions in existing counts,
which stayed the same).

## What this does not do

- The slippage/spread/volatility guardrails only ever *reject* an order —
  none of them modify, resize, or split an order to fit under a
  threshold. A real reduce-only order type (one Kraken clips server-side
  rather than rejects outright) was considered and not used here, since
  this project's `AddOrderRequest` has no such flag wired through yet;
  the volatility breaker's freeze is enforced entirely on the reject/
  approve boundary in `risk.rs`, before an order ever reaches
  `kraken_rest.rs`.
- The rate limiter's per-call cost is flat, not Kraken's real per-endpoint
  cost table (AddOrder/CancelOrder's actual cost depends on order
  lifetime in Kraken's real model, which this doesn't attempt to
  replicate).
- None of this has been exercised against a live Kraken account or a real
  flash-crash-shaped market — same caveat as the rest of this project's
  execution path (see `kraken_rest.rs`'s top-of-file docs).
