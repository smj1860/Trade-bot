# Model training: feature parity between live and historical data

Follow-up to `docs/historical-data-pipeline.md`, once the pipeline was
actually producing real historical OHLC data in Supabase and the next
question became "okay, now how do we train a model on it." Decided with
Stephen on 2026-09-26.

## The gap

The live Python strategy layer's tick-level features (`strategy/features.py`,
`Features.imbalance` and `Features.spread`) are computed from the **live
order book's best bid/ask sizes** — how much size is resting at the top of
book right now. Kraken's historical OHLC candles (what `historical-data/`
backfills) only ever recorded **open/high/low/close/volume/trade-count** per
bar — no book-depth information at all. There is no way to reconstruct
`imbalance` or a faithful `spread` history from OHLC data; only something
`momentum`-shaped (based on price changes between candles) can be
reconstructed.

That meant a model trained today could either:
1. Train on `momentum` alone — a much weaker feature set than what the live
   engine could in principle compute, or
2. Wait until enough *live* order-book depth has been logged to train on the
   real tick-level features, which could be weeks/months away and produces
   nothing usable now, or
3. **Extend the live engine with new features that are computed identically
   from bar closes, both live and historically** — more upfront engineering,
   but a model can be trained today against real Kraken history, on features
   the live engine can actually reproduce tick-for-tick.

Stephen chose option 3.

## Design: one indicator implementation, two callers

`python-strategy/strategy/indicators.py` holds four pure functions —
`sma_ratio`, `rsi`, `realized_vol`, `bar_momentum` — each taking a plain
ordered list of bar closes (oldest first) and returning a float. They have
no notion of "live" or "historical," no I/O, no state. That's deliberate:
the same functions are called from two very different places —

- **Live**: `strategy/bars.py`'s `BarAggregator` buckets live mid-price
  ticks into fixed-duration bars (aligned to epoch, same duration as the
  historical candles — see `bar_interval_minutes` below) and hands a
  rolling window of completed closes to these functions on every tick, via
  `strategy/features.py`'s `FeatureEngine`.
- **Historical/training**: the training script (once written — see "Not yet
  built" below) will query `ohlc_candles` from Supabase and hand a rolling
  window of `close` values from that table to the *exact same* functions.

Whatever a model learns to weight during training is the same computation
the live engine runs — there's no second implementation to silently drift
out of sync with the first.

## What changed

- **`strategy/indicators.py`** (new): the four pure functions above, plus
  `sma`. Every function returns `0.0` — a neutral "no opinion" value — when
  there isn't yet enough history, rather than raising.
- **`strategy/bars.py`** (new): `BarAggregator`, buckets `(timestamp, price)`
  ticks per symbol into bars and keeps a bounded rolling history of
  completed closes. Bars are bucketed on **event time** (`floor(timestamp /
  bar_interval_seconds) * bar_interval_seconds`), not tick count, so a bar
  here means the same thing as an hourly/daily OHLC candle from Kraken.
  Only completed bars are ever exposed — the still-forming current bucket
  never leaks into the window, so live and training never see a
  differently-shaped input.
- **`strategy/features.py`**: `Features` gained four new fields —
  `sma_ratio`, `rsi`, `realized_vol`, `bar_momentum` — all defaulting to
  `0.0` so existing test/call sites that construct a `Features` without
  them keep working. `FeatureEngine` now also maintains a `BarAggregator`
  internally and accepts an optional `timestamp` on
  `on_order_book_update` (falls back to wall-clock time when omitted).
- **`strategy/base.py` / `strategy/imbalance_momentum.py`**: `Strategy.
  on_order_book_update`'s signature gained an optional, keyword-only
  `timestamp` parameter, threaded through to the feature engine.
- **`strategy/engine.py`**: now passes `update.exchange_timestamp_ns`
  (converted to seconds) through as that timestamp, so bars are built on
  the exchange's own event time rather than local processing time — and
  logs the four new feature values alongside the existing ones.
- **`strategy/models.py`**: both `SklearnModelWrapper._feature_vector` and
  `TorchModelWrapper._feature_vector` now expose the four new feature names,
  so a `feature_order` in `strategy_config.toml` can reference them once a
  real trained model exists.
- **`strategy/config.py` / `strategy_config.example.toml`**: new
  `[strategy.features]` keys — `bar_interval_minutes` (default `60`),
  `sma_window` (`20`), `rsi_window` (`14`), `vol_window` (`20`),
  `bar_momentum_window` (`10`). **`bar_interval_minutes` must match
  whatever `interval_minutes` the training data is pulled at** — the
  defaults line up with `historical-data/backfill_ohlc.py`'s 60-minute
  candles, which is what's actually been backfilled with real depth so far
  (~1 month of hourly history per symbol as of this writing).
- Tests: `tests/test_indicators.py` and `tests/test_bars.py` (new), plus
  additions to `tests/test_features.py` and `tests/test_models.py`. Full
  suite (63 tests) passes.

## A known approximation

Live bars are built from **mid-price ticks** (best-bid/best-ask midpoint),
because the strategy layer doesn't currently see individual trades or
volume — only order-book updates. Historical bars are built from Kraken's
real traded OHLC candles. A bar's *close* is therefore "last mid-price tick
before the bucket rolled over" live, vs. "last traded price in that minute"
historically. These are usually close but not identical — mid-price can sit
slightly off the last trade, especially in a fast-moving or thin book. This
is a real, acknowledged gap, not a silent one: if a trained model's live
performance meaningfully diverges from its backtest, this is one of the
first places to look. Fixing it properly would mean wiring a real trade
feed into the strategy layer (Rust core would need to forward trades, not
just book updates) — worth doing later if it turns out to matter, not done
now because there's no live evidence yet that it does.

## Not yet built

- The actual training script: query `ohlc_candles` for a chosen symbol at
  60-minute resolution, compute `sma_ratio`/`rsi`/`realized_vol`/
  `bar_momentum` via `strategy.indicators` over rolling windows, define a
  label (e.g. next-bar-direction), do a time-ordered (non-shuffled)
  train/test split, train a simple sklearn classifier, evaluate against a
  naive baseline, save via `joblib`.
- Documenting the resulting `feature_order` once a model actually exists.
