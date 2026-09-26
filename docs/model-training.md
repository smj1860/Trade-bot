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

## The training script (`python-strategy/scripts/train_model.py`)

Queries `ohlc_candles` for one symbol/interval, computes the four
bar-derived features over rolling windows using the exact same functions
`strategy/features.py` calls live, labels each row with next-bar direction
(a deliberately naive baseline label — see the script's own docstring for
why), does a time-ordered train/test split (never shuffled, to avoid
lookahead leakage), trains a baseline `sklearn` classifier
(`LogisticRegression` by default, or `GradientBoostingClassifier` via
`--kind gboost`), evaluates it against two naive baselines (majority-class
and "the last move persists"), and saves the model via `joblib`.

```
export SUPABASE_DB_URL=postgresql://...   # Session pooler string, see historical-data-pipeline.md
python3 scripts/train_model.py --symbol BTC-USD
```

Needs `requirements-training.txt` (`psycopg2-binary`) alongside
`requirements-ml.txt` (`scikit-learn`, `joblib`).

### First real run, against actual Supabase data (2026-09-26)

Ran against BTC-USD's real 723 hourly candles (the full history backfilled
so far, spanning late Aug through Sept 2026):

| | value |
|---|---|
| dataset rows (after warmup) | 702 |
| train / test split | 561 / 141 (time-ordered, last 20%) |
| positive-label rate | 51.3% (close to coin-flip) |
| **model accuracy** | **0.489** |
| majority-class baseline | 0.532 |
| persistence baseline | 0.482 |

**The model did not beat the majority-class baseline.** This is an honest
negative result, not a bug: one month of hourly BTC-USD candles for one
symbol is a small, noisy dataset, next-bar direction is a genuinely hard
target (BTC-USD hourly moves are close to a random walk at this
resolution), and `LogisticRegression` with 4 simple technical features is
a deliberately minimal starting point. This does **not** mean the
feature-parity engineering was wasted — it means the honest next step is
more/better data and a more considered label, not treating this model as
tradeable. **This trained model is not committed to the repo** (`.gitignore`
now excludes `python-strategy/models/`) precisely because it isn't a
validated result — regenerate it from real Supabase data with the command
above rather than trusting a stale binary.

## Second round: EMA, Bollinger Bands, Awesome Oscillator (2026-09-26)

Added three more bar-derived indicators to `strategy/indicators.py`, following
the same pure-function/no-state pattern as the first four:

- **`ema` / `ema_ratio`** — exponential moving average and the EMA-based
  analog of `sma_ratio` (price vs. its EMA).
- **`bollinger_percent_b` / `bollinger_bandwidth`** — where price sits
  relative to its Bollinger Bands (rescaled to roughly `[-1, 1]`, 0 = at
  the middle band, unlike the traditional `[0, 1]` %b, so it composes with
  this project's other signals) and how wide the bands currently are (a
  volatility feature, distinct from `realized_vol`'s log-return-based
  measure).
- **`awesome_oscillator`** — Bill Williams' classic 5/34-period SMA
  difference, computed over bar *midpoints* (`(high + low) / 2`), not
  closes — the one indicator here that needs more than a close per bar.

That last point required extending `strategy/bars.py`'s `BarAggregator` to
also track each bar's high/low (from the tick range seen within that
bucket) and expose a parallel `midpoints()`/`midpoint_window()` history
alongside the existing `closes()`/`window()`. Historically, this is
actually *better* than the live approximation: `scripts/train_model.py`
computes AO from Kraken's own real recorded high/low per candle, while the
live engine can only approximate a bar's high/low from whatever mid-price
ticks it happened to see in that bucket — a known, documented asymmetry
between the two (see `bars.py`'s docstring), not a silent one.

`Features`, `FeatureEngine`, `models.py`'s wrappers, `config.py` +
`strategy_config.example.toml`, and `scripts/train_model.py` were all
extended the same way as the first round: new fields/config keys with
sensible defaults, nothing existing broken. 22 new tests (85 total, all
passing). Ran the extended training script against BTC-USD's real 723
hourly candles with all 8 features: 0.522 model accuracy vs. 0.529
majority-class / 0.486 persistence baseline — still not a clear win, same
honest conclusion as the first round (see below).

### Indicators considered but not (yet) added

A few other well-known technical indicators, and why they're not here:

- **MACD** (moving average convergence/divergence) — a natural next
  addition; essentially a difference of two EMAs plus a signal-line EMA of
  that difference. Not added yet only because nothing has asked for it —
  same `indicators.py` pattern would fit it directly.
- **Stochastic Oscillator** — needs high/low per bar (now available via
  `bars.py`'s midpoint tracking, or real candle high/low historically),
  same shape as Awesome Oscillator. Not yet added.
- **ATR (Average True Range)** — a volatility measure like
  `bollinger_bandwidth`/`realized_vol`, but computed from true range
  (accounts for gaps between bars), which needs the *previous* bar's close
  as well as the current bar's high/low. Not yet added.
- **ADX (Average Directional Index)** — a trend-strength indicator built
  on top of directional movement + ATR; meaningfully more involved to
  implement correctly than anything here so far. Not added.
- **CCI (Commodity Channel Index)**, **Williams %R** — both similar in
  spirit to Bollinger %b/RSI (price relative to a recent range); would be
  quick additions in the same pattern if wanted.
- **OBV (On-Balance Volume)**, **VWAP** — both need real traded volume
  attributed to price direction. Kraken's OHLC candles *do* carry a volume
  column (unused so far), but the live engine has no live volume signal at
  all (only order-book ticks) — these would need the same kind of
  live/historical parity thinking the rest of this doc is about, and
  aren't started.
- **Parabolic SAR**, **Ichimoku Cloud** — more elaborate, multi-line
  indicators; skipped for now since nothing built here needs that level of
  sophistication yet, and simpler indicators haven't shown signal.

### Not yet done / natural next steps

- More history: 723 hourly candles is thin for training; either let the
  scheduled backfill keep accumulating, or pull Kraken's own deeper
  historical CSV dumps via `import_csv.py` (see
  `docs/historical-data-pipeline.md`'s "Bulk CSV import" section).
- Try daily (1440-minute) candles too — more history per candle, a
  possibly less noisy target, at the cost of far fewer completed bars per
  unit time.
- A better label than raw next-bar direction (e.g. a magnitude threshold,
  or a multi-bar-ahead horizon) — next-bar direction is the simplest
  possible thing to try first, not the right final target.
- Multi-symbol training (pool all 13 symbols' candles into one dataset)
  rather than one model per symbol from ~700 rows each.
- Once a model actually beats its baselines convincingly, wire
  `feature_order` into `strategy_config.toml` and switch
  `strategy.model.kind` to `"sklearn"` — not before.
