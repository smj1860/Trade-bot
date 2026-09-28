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

## Third round: MACD, CCI, Williams %R (2026-09-26)

Added three more bar-derived indicators to `strategy/indicators.py`, same
pure-function/no-state pattern as the first two rounds:

- **`_ema_series` (new private helper) / `macd_histogram`** — `ema()`
  treats a window's own length as its EMA period, which doesn't work for
  MACD: MACD's fast/slow EMAs need *fixed* periods (classically 12/26)
  regardless of how large a window of closes is fed in, and its signal
  line is an EMA of the resulting MACD series, not of raw closes. So
  `_ema_series(values, period)` computes a full iterative EMA series (one
  value per input) at a fixed period, and `macd_histogram` builds the
  fast/slow EMA series, takes their difference as the MACD line, EMAs
  *that* to get the signal line, and returns `(macd - signal)` at the most
  recent point, normalized by the current close for cross-symbol
  comparability.
- **`cci`** — Commodity Channel Index: how far the most recent typical
  price ((high+low+close)/3) sits from its SMA, relative to the window's
  mean absolute deviation, rescaled by /100 (traditional +-100
  overbought/oversold threshold) to match this project's [-1, 1]
  composability convention.
- **`williams_percent_r`** — where the most recent close sits within the
  window's high-low range, rescaled from the traditional [-100, 0] to
  roughly [-1, 1] (the same rescale idiom as RSI).

CCI and Williams %R both need real per-bar high/low history, not just
their average (which `midpoints` already provided for the Awesome
Oscillator) — `strategy/bars.py`'s `BarAggregator` was extended with
`highs`/`lows` deques (populated from the already-tracked
`current_high`/`current_low` at bar completion) and matching
`highs()`/`lows()`/`high_window()`/`low_window()` accessors, mirroring the
existing `closes()`/`window()` pattern. `FeatureEngine` folded all its
window-sizing logic (previously a two-tier `max_bars`/`max_ao_bars` split)
into one `max()` across every indicator's required window, since the
underlying bar histories all share one `maxlen` anyway.

`Features`, `models.py`'s wrappers, `config.py` +
`strategy_config.example.toml`, `engine.py`'s signal logging, and
`scripts/train_model.py` (including `load_ohlc`, which now returns
highs/lows alongside closes/midpoints) were all extended the same way as
prior rounds. 25 new tests (110 total, all passing).

Ran the extended training script against BTC-USD's real 723 hourly
candles with all 11 features:

| | value |
|---|---|
| dataset rows (after warmup) | 688 |
| train / test split | 550 / 138 (time-ordered, last 20%) |
| positive-label rate | 51.5% |
| **model accuracy** | **0.529** |
| majority-class baseline | 0.529 |
| persistence baseline | 0.486 |

**The model exactly tied the majority-class baseline** — still not a
clear win. Adding more indicators hasn't moved the needle on this
dataset/label combination; three rounds in a row point the same
direction (see "Not yet done / natural next steps" below): the limiting
factor looks like it's the data/label, not the feature set. This is the
signal to try pooling more symbols and/or a better label next, rather
than continuing to add indicators one at a time.

## Multi-symbol pooling (2026-09-26)

`scripts/train_model.py --symbol` now also accepts a comma-separated list
("BTC-USD,ETH-USD,SOL-USD") or the literal `all` (every symbol with data
at the given `--interval`, discovered with a `select distinct symbol`
query rather than importing `historical-data/`'s `symbols.py` — this
script stays intentionally standalone). Both pool multiple symbols'
history into one combined training set, instead of training a separate
model per symbol from ~700 rows each.

Every bar-derived feature here was already a ratio or a value rescaled to
roughly `[-1, 1]` — a deliberate choice from the first round on, so a
symbol at $80,000/BTC and one at $1/DOGE produce comparable numbers — so
pooling is a matter of computing each symbol's features independently
over its own close/high/low history (no cross-symbol math anywhere) and
combining the resulting rows. What still had to be handled carefully:
each symbol is time-ordered split *individually* before any pooling
happens, then every symbol's train rows are concatenated together and
every symbol's test rows are concatenated together — never split the
pooled rows as one long sequence, which would let one symbol's split
boundary land in the middle of another symbol's history and leak
lookahead across a symbol boundary that time itself never crosses.
Baselines are pooled the same way: majority-class over the pooled
training labels, and persistence accuracy aggregated as
total-correct/total-bars across every symbol's own held-out period (not
an average of per-symbol accuracies, so a symbol with more test bars
counts proportionally more). A symbol with too little history to clear
warmup is skipped with a warning rather than aborting the whole run.

New tests in `tests/test_train_model.py` cover `resolve_symbols` (single
symbol, comma-list, `all`), the warmup-skip behavior, and — the important
one — that pooling's per-symbol-then-concatenate approach never lets a
row cross from one symbol's train split into another symbol's test split
(117 tests total, all passing).

Ran it for real against all 13 symbols' real Supabase history (723 hourly
candles each, fetched live via the Supabase MCP tool rather than
`SUPABASE_DB_URL`, which isn't set in this sandbox):

| | value |
|---|---|
| symbols pooled | 13 (AAVE, AVAX, BTC, DOGE, ETH, LINK, NEAR, PENDLE, SOL, SUI, TAO, UNI, XRP, all -USD) |
| dataset rows (after warmup, pooled) | 8,944 |
| train / test split | 7,150 / 1,794 (each symbol time-split individually, then concatenated) |
| positive-label rate (train) | 50.4% |
| **model accuracy** | **0.514** |
| majority-class baseline | 0.513 |
| persistence baseline | 0.484 |

**Still not a clear win** — the model barely edges the majority-class
baseline (0.514 vs 0.513, essentially noise) — but this is the most
informative negative result yet: going from ~700 rows (one symbol) to
~9,000 rows (13 symbols pooled) moved the positive-label rate to almost
exactly 50/50 (a genuinely harder, less exploitable target) without the
model gaining any real edge over guessing the majority class. That's
consistent with next-bar direction on hourly crypto candles being close
to a random walk regardless of how much more data or how many more
technical indicators get thrown at it — the next thing worth trying is a
different label (a magnitude threshold, or a multi-bar-ahead horizon),
not more data or more indicators in this same shape.

## Label engineering: horizon + magnitude threshold (2026-09-26)

Kraken's spot taker fee at the entry volume tier is 0.80% — a round trip
(in + out) costs at least 1.6% before spread/slippage, so "did price go up
or down" is the wrong question for a 1-hour-ahead label anyway: a bar that
moves 0.1% and gets called "up" is not a trade worth making even if the
label is technically correct. `scripts/train_model.py` gained two label
knobs, combinable:

- **`--horizon N`** — label bar `i` by the direction of the move to bar
  `i+N`, instead of always `i+1`. Longer horizons give price more room to
  make a real move (at the cost of overlapping, correlated label windows).
- **`--min-move X`** / **`--top-fraction F`** — drop any row whose move
  over the horizon doesn't clear a fixed fraction (`--min-move`) or the
  most-extreme-`F`-fraction cutoff computed from *that symbol's own* move
  distribution (`--top-fraction`, so the threshold means the same thing
  for calm BTC and choppy DOGE instead of one flat percentage). When both
  are given, whichever is larger for that symbol wins.

The persistence baseline was updated to match: it now predicts the same
direction as the most recent completed *horizon-length* move, and is
scored only on the exact rows the min-move/top-fraction filtering kept —
otherwise a "which baseline is stronger" comparison would silently be
comparing accuracy on two different sets of bars. `build_dataset` now
also returns each kept row's bar index, so a pooled run can score the
baseline correctly per symbol even after filtering shrinks each symbol's
row count differently. 9 new tests (126 total, all passing).

Ran it for real, against the same pooled 13-symbol dataset (fetched live
via the Supabase MCP tool), sizing the threshold above the actual 1.6%
round-trip cost with a profit margin (4.5%), at two horizons:

| | 1h (baseline) | 12h | 24h |
|---|---|---|---|
| threshold | none | 4.5% (or top 30%, per symbol) | 4.5% (or top 30%, per symbol) |
| pooled rows | 8,944 | 1,173 | 2,049 |
| train / test | 7,150 / 1,794 | 933 / 240 | 1,638 / 411 |
| positive-label rate | 50.4% | 77.7% | 77.0% |
| **model accuracy** | 0.514 | 0.625 | 0.650 |
| majority-class baseline | 0.513 | **0.637** | **0.657** |
| persistence baseline | 0.484 | 0.467 | 0.399 |

**Filtering to big moves didn't create an easier problem — it created a
skewed one.** Over this particular ~1-month window, most moves large
enough to clear 4.5% happened to be *up* moves (77% positive-label rate,
vs. the roughly 50/50 split at 1h), which makes "always predict up" a
strong baseline almost by definition — 0.637–0.657 accuracy just from
guessing the majority class every time. The model still doesn't beat that
strengthened baseline at either horizon. Two honest caveats, not spin:
this dataset is one month of a market that was mostly trending up over
that stretch, so the 77% skew is a property of *this sample window*, not
necessarily a durable fact about crypto; and filtering shrinks the dataset
hard (roughly 9,000 rows down to ~1,200–2,000), which is its own separate
reason results here carry less statistical weight than the unfiltered
runs. The label change was worth making — it's the economically correct
question to ask — but by itself, with only one month of history, it
hasn't produced a model that's actually tradeable either.

### Indicators considered but not (yet) added

A few other well-known technical indicators, and why they're not here:

- **Stochastic Oscillator** — needs high/low per bar (now available via
  `bars.py`'s high/low tracking, or real candle high/low historically),
  same shape as Awesome Oscillator. Not yet added.
- **ATR (Average True Range)** — a volatility measure like
  `bollinger_bandwidth`/`realized_vol`, but computed from true range
  (accounts for gaps between bars), which needs the *previous* bar's close
  as well as the current bar's high/low. Not yet added.
- **ADX (Average Directional Index)** — a trend-strength indicator built
  on top of directional movement + ATR; meaningfully more involved to
  implement correctly than anything here so far. Not added.
- **OBV (On-Balance Volume)**, **VWAP** — both need real traded volume
  attributed to price direction. Kraken's OHLC candles *do* carry a volume
  column (unused so far), but the live engine has no live volume signal at
  all (only order-book ticks) — these would need the same kind of
  live/historical parity thinking the rest of this doc is about, and
  aren't started.
- **Parabolic SAR**, **Ichimoku Cloud** — more elaborate, multi-line
  indicators; skipped for now since nothing built here needs that level of
  sophistication yet, and simpler indicators haven't shown signal.

### Net P&L / simulated paper-trade label (2026-09-26)

The horizon+threshold round above sized the label's magnitude threshold at
a flat, chosen number (4.5%, "cover the round-trip cost plus a profit").
The natural next question — "what about running training on paper trades?"
— split into two different things:

1. **Live/forward paper trading**: extend `engine.py`'s `dry_run_only`
   path to simulate fills and accumulate real P&L over weeks/months. Real,
   but slow, and only reflects whatever policy generates the trades.
2. **Offline net-P&L label refinement** (this round): keep using existing
   historical OHLC data, but stop treating "--min-move" as an arbitrary
   number and instead **derive it directly from Kraken's real trading
   costs**, and evaluate models on **simulated net P&L** (what an actual
   paper trade would have made after real round-trip costs), not just
   classification accuracy. No waiting required — this is computed
   directly over history that already exists.

Concretely, in `scripts/train_model.py`:

- `--taker-fee` (default 0.008 = Kraken's entry-tier 0.80% taker fee) and
  `--profit-margin` (default 0.0) combine into a round-trip cost
  (`2 * taker_fee + profit_margin`), which is now `--min-move`'s **default**
  whenever `--min-move` is omitted (its new default is `None`, meaning
  "derive it," rather than an arbitrary flat number like the 4.5% used
  above). An explicit `--min-move` still overrides this.
- `net_pnl(closes, i, horizon, predicted_up, round_trip_cost)` computes one
  simulated trade's net return: the actual directional move (inverted if
  the call was "down"/short) minus the round-trip cost.
  `simulate_net_pnl(...)` aggregates this across a set of (index,
  prediction) pairs, returning `(total, count)` so results can be summed
  across pooled symbols before dividing into a mean.
- `main()` now prints simulated net P&L per trade (and the total over the
  test period) for the trained model, the majority-class baseline, and the
  persistence baseline — not just their classification accuracy — because
  a model can have higher accuracy than a baseline and still lose more
  money per trade if its correct calls are on smaller moves and its wrong
  calls are on larger ones.

**Ran it for real**, against the same pooled 13-symbol dataset (723 hourly
candles/symbol), at the fee-derived default threshold (1.6% = 2×0.8%
taker fee, no profit margin) and a couple of variations, for comparison:

| | 12h, fee-derived (1.6%) | 12h, fee-derived + 1% margin (2.6%) | 24h, fee-derived (1.6%) | 12h, flat 4.5% (prior round) |
|---|---|---|---|---|
| test rows | 842 | 542 | 1,127 | 245 |
| model accuracy | 0.552 | 0.576 | 0.618 | 0.624 |
| majority-class accuracy | 0.637 | 0.625 | 0.643 | 0.637 |
| **model net P&L/trade** | -0.0155 | -0.0096 | -0.0030 | +0.0045 |
| **majority net P&L/trade** | -0.0043 | -0.0021 | **+0.0036** | +0.0066 |
| **persistence net P&L/trade** | -0.0166 | -0.0194 | -0.0197 | -0.0248 |

**Honest read: net P&L is a harsher, more informative test than accuracy,
and by that test nothing here is tradeable yet.** At the fee-derived
1.6%/2.6% thresholds, *every* strategy — model and both baselines — loses
money per trade on average, because most rows just barely clear the
threshold (a move a little over 1.6% still routinely reverses or gives
back the edge within the horizon) — moving the threshold up to 24h/1.6%
gets the majority-class baseline barely net-positive (+0.0036/trade), and
only the previously-tried, much stricter 4.5% flat threshold (which throws
away most of the data down to 245 test rows) got the model itself
net-positive, and even there the majority-class baseline still edges it
out. This confirms two things at once: the net-P&L metric is doing its
job (accuracy alone was hiding real economics — a model with 0.552-0.624
accuracy can still lose money after costs), and the underlying problem
from every round so far persists — one month of mostly-one-direction data
isn't enough to find a threshold/horizon combination where a *trained*
model actually beats naive baselines on real money, not just direction
calls. The fee-derived default is still the economically correct place to
start (a threshold below real costs was always going to produce
unprofitable "correct" calls by definition), but validating this
approach for real needs the same thing every prior round has been
missing: more history spanning more than one trend regime.

### Not yet done / natural next steps

- More history: 723 hourly candles is thin for training; either let the
  scheduled backfill keep accumulating, or pull Kraken's own deeper
  historical CSV dumps via `import_csv.py` (see
  `docs/historical-data-pipeline.md`'s "Bulk CSV import" section).
- Try daily (1440-minute) candles too — more history per candle, a
  possibly less noisy target, at the cost of far fewer completed bars per
  unit time.
- More history above all else: every round so far — three of indicators,
  one of pooling, one of label engineering — has run into the same wall,
  ~1 month of hourly candles per symbol. The label-engineering round
  specifically showed that a magnitude-thresholded label needs enough
  history that "moves big enough to clear the threshold" isn't dominated
  by whichever direction happened to trend during that one month. Without
  more history, `--horizon`/`--min-move`/`--top-fraction` can be re-run
  as data accumulates, but a truly different verdict is unlikely from the
  same month of data sliced a different way.
- Once real trending-vs-ranging periods are represented (which needs
  multiple months, not one), the label-engineering approach here should
  be revisited — it's still the economically correct question (a label
  that ignores trading costs was always going to be an odd thing to
  train on), just underpowered on a month of mostly-one-direction data.
- Once a model actually beats its baselines convincingly, wire
  `feature_order` into `strategy_config.toml` and switch
  `strategy.model.kind` to `"sklearn"` — not before.

### Deep-history horizon sweep with walk-forward + triple-barrier (2026-09-27)

With all 14 symbols now carrying a full 6-month deep backfill (via
`historical-data/backfill_ohlc_from_trades.py`) and both `--folds`
(expanding-window walk-forward validation) and `--label-scheme
triple-barrier` built in `scripts/train_model.py`, the natural next step
was to actually run the sweep the earlier rounds kept saying was blocked
on "more history."

Ran `--symbol all --label-scheme triple-barrier --folds 5 --kind gboost`
across `--horizon` ∈ {4, 8, 12, 24, 48} bars (hourly candles), via the new
`train-model.yml` GitHub Actions workflow (added this round so
`SUPABASE_DB_URL` never has to leave the repo's secrets — see that
workflow's own docstring). Each horizon's barrier width is the same
fee-derived threshold used throughout this doc (round-trip taker fee +
slippage, no profit margin), so a "win" nets roughly breakeven and a
"loss" nets roughly `-2 × barrier_pct` — see `triple_barrier_net_pnl`.

| horizon (bars) | avg accuracy | vs majority / persistence | avg net P&L/trade | vs majority / persistence | folds beating both baselines (of 5) |
|---|---|---|---|---|---|
| 4 | 0.513 | 0.479 / 0.501 | -0.0166 | -0.0177 / -0.0170 | 2 |
| 8 | 0.495 | 0.467 / 0.509 | -0.0172 | -0.0181 / -0.0167 | 1 |
| 12 | 0.482 | 0.474 / 0.519 | -0.0176 | -0.0179 / -0.0164 | 0 |
| 24 | 0.477 | 0.484 / 0.514 | -0.0178 | -0.0176 / -0.0165 | 0 |
| 48 | 0.482 | 0.490 / 0.501 | -0.0176 | -0.0173 / -0.0170 | 1 |

**Honest read: none of these clear the bar.** Horizon=4 is the closest —
its *averaged* accuracy and net P&L both edge out both baselines — but it
only actually beats both baselines, on both metrics, in 2 of 5 folds. An
average that looks good only because a couple of folds carried it is
exactly the "looks good on one time period" failure mode this doc has
flagged before; it is not a validated result, and none of the five
horizons pass "beats both baselines on both metrics, consistently."

Net P&L being negative for the model *and both baselines*, in every fold,
at every horizon, is also worth being straight about: because each
barrier is sized at the fee-derived breakeven threshold (no profit
margin), a correct call nets roughly 0 before slippage noise and an
incorrect call nets roughly `-2 × barrier_pct` — so even a "good" model
here is fighting to lose less, not actually turning a profit, by
construction. That's a labeling-threshold artifact, not evidence the
underlying signal is hopeless; a real profit-margin sweep (`--profit-margin
0.01` etc., same as the earlier flat-4.5%-threshold experiment showing a
sign of life) hasn't been tried on the full deep-history pooled dataset
yet and is the more informative next experiment than more horizons at
breakeven.

**Verdict: no model saved or wired in this round.** Per the standing rule
above, `feature_order`/`strategy.model.kind` stay untouched until a
configuration actually earns it.

Natural next steps, in the order they'd actually move the needle:
1. Re-run the sweep with `--profit-margin` > 0 (e.g. 0.01, 0.02) on the
   full deep-history pooled dataset — the flat-4.5%-threshold result from
   the prior round (before deep history existed) was the one time a model
   went net-positive, and margin is the parameter that recreates that
   condition properly (fee-derived, not an arbitrary flat number).
2. Sweep `--min-move`/`--top-fraction` independently of horizon now that
   there's enough history per symbol to not immediately starve the test
   set.
3. Re-run per-symbol (not just pooled) now that 6 months exists per
   symbol — pooling assumes one feature/threshold combination generalizes
   across very different coins, which hasn't been checked against the
   deep-history data yet.

### Cross-asset pooling review, volume features, and purged walk-forward CV (2026-09-27)

Stephen proposed three ML-engineering improvements aimed at making
cross-asset pooling sound: (1) train a single pooled model, or 2-3
cluster models (majors vs. mid-cap alts), instead of 14 independent
per-pair models; (2) normalize every feature so raw prices/volumes never
enter the model, since 14 pairs span very different price/volume levels;
(3) purge/embargo walk-forward CV fold boundaries so overlapping
triple-barrier label windows can't leak between train and test.

Went through each honestly rather than assuming they all needed new code:

1. **Pooling/clustering — already free, no code needed.** `--symbol`
   already accepts a comma-separated list or `all`
   (`resolve_symbols()`), so a "majors" cluster and an "alts" cluster are
   just two different `--symbol` values to the existing script (e.g.
   `--symbol BTC-USD,ETH-USD` vs. every other symbol) — not a feature
   that needed building. Worth running as an experiment, but it's a
   sweep parameter, not an engineering gap.

2. **Normalization — already true, checked rather than assumed.** Read
   every function in `strategy/indicators.py`: `sma_ratio`/`ema_ratio`
   (ratio to own average), `rsi`/`williams_percent_r` ([-1, 1]-rescaled),
   `realized_vol` (std of *log* returns, already scale-free),
   `bollinger_percent_b`/`bollinger_bandwidth` (normalized by std/middle
   band), `macd_histogram` (explicitly divided by the current close),
   `cci` (normalized by mean absolute deviation), `awesome_oscillator`
   (normalized by the slow SMA) — every one is already a dimensionless
   ratio, never a raw price. This was good news, not a gap: the concern
   was legitimate in general (pooling raw prices across BTC and a
   sub-$1 altcoin would be a real bug), it just turned out to already be
   handled.

   What genuinely *was* missing: **no volume-derived feature existed at
   all**, despite `ohlc_candles.volume` being recorded all along and
   `strategy/bars.py`'s `BarAggregator` already tracking real per-bar
   volume (from this session's earlier trade-tick-ingestion work).
   Added two new ratio-based indicators to `strategy/indicators.py`:
   - `volume_ratio(volumes)`: (most recent bar's volume / SMA of the
     window) - 1 — the same self-relative-ratio idiom as
     `sma_ratio`/`ema_ratio`, applied to volume instead of price, so a
     low-cap altcoin trading 3x its own average and BTC trading 3x its
     own (much larger) average read identically.
   - `parkinson_vol(highs, lows)`: Parkinson's (1980) high-low range
     volatility estimator — `sqrt(mean(ln(high/low)^2) / (4 ln 2))` — a
     second, independent volatility read alongside `realized_vol`'s
     close-to-close estimate; a bar that spiked hard in both directions
     before closing flat looks calm to `realized_vol` but clearly
     volatile here. Already scale-free (a ratio of prices), same as
     `realized_vol`.

   Both wired into `strategy/features.py`'s `FeatureEngine` (reusing its
   existing `vol_window`, no new constructor parameter) and appended to
   `scripts/train_model.py`'s `FEATURE_ORDER` (appended, not interleaved,
   so an existing `feature_order` in `strategy_config.toml` pointing at
   the first 11 features stays meaningful rather than silently
   reordered). `load_ohlc()` now selects `volume` from `ohlc_candles`
   alongside close/high/low. Both features correctly read as neutral
   (0.0 / near-0.0) for a symbol whose live bars are only ever fed via
   `on_tick()` (no real trade feed wired up yet for that symbol) — same
   "no opinion" convention every other indicator already follows.

3. **Purged + embargoed walk-forward CV — a real, fixed gap, in *two*
   places.** `walk_forward_splits()` had no purging: a training row near
   a fold boundary could have a label computed from bars that fall
   inside that fold's test block (a fixed-horizon label looks to bar
   `i+horizon`; a triple-barrier label can touch a barrier anywhere in
   `i+1..i+horizon`), which is literal label leakage across the
   boundary — exactly the risk Stephen flagged. Implemented the standard
   purge-then-embargo recipe (Lopez de Prado): at each fold boundary,
   purge trailing training rows whose label lookahead reaches into the
   test block, and embargo leading test rows within the same buffer of
   the boundary (protects against serial correlation across it, not just
   literal overlap). New `--embargo N` CLI arg, defaulting to `None` =
   each symbol's own `--horizon` (the tightest correct minimum — the
   label's actual maximum forward reach).

   **The same leakage existed in `main()`'s single time-ordered split
   too** — the path that actually trains and saves the production
   model, not just the walk-forward validation tool. It's the identical
   bug, just less obviously connected to "CV fold boundaries." Fixed it
   with the same shared `_purge_train_end`/`_embargo_test_start` helper
   functions rather than leaving the evaluation path leakage-free while
   the production training path silently wasn't — an inconsistency that
   would have been worse than not fixing either.

   Purging is computed in real bar-index space (via each row's `indices`
   entry), not row-count space, since `--min-move`/triple-barrier
   filtering already leaves gaps between kept rows — a fixed row-count
   purge would under- or over-purge depending on how much filtering
   happened to remove near a given boundary.

185 Python tests total (up from 165), 60 Rust tests, all passing.
Nothing here changes the sweep verdict above (no model saved yet) — this
was tightening the training pipeline's correctness ahead of the next
real sweep, not a new sweep result. The next sweep (profit-margin, or a
re-run of the horizon sweep now that leakage is closed and volume
features exist) should be run against this corrected pipeline, not the
prior one.
