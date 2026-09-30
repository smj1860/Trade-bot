#!/usr/bin/env python3
"""
Trains a baseline classifier on the OHLC-derived bar features
(strategy/indicators.py: sma_ratio, ema_ratio, rsi, realized_vol,
bar_momentum, bollinger_percent_b, bollinger_bandwidth, awesome_oscillator,
macd_histogram, cci, williams_percent_r, volume_ratio, parkinson_vol,
returns_zscore — see FEATURE_ORDER) using real historical Kraken candles
from Supabase,
and saves it via joblib for use as strategy.model.kind = "sklearn" (see
strategy_config.example.toml). Every feature is already a dimensionless
ratio (a fraction of price, of the feature's own recent average, or of a
[-1, 1]-rescaled range) rather than a raw price or volume — deliberately,
so pooling symbols at wildly different price/volume levels (--symbol all,
or a manually chosen subset, e.g. "BTC-USD,ETH-USD" for a majors-only
pool vs. the rest for a mid-cap-alt pool — no separate cluster-model
machinery needed, this is just --symbol with a different comma list)
never lets one symbol's absolute scale dominate what the model learns.

This is the other half of the feature-parity work described in
docs/model-training.md: strategy/features.py's FeatureEngine computes
these same features live, from the same pure functions in
strategy/indicators.py, over bars built by strategy/bars.py's
BarAggregator. This script computes them the same way over completed
windows of real historical closes (and, for the Awesome Oscillator,
Kraken's own real historical high/low per candle — see
awesome_oscillator's use of load_ohlc below, which is actually a slightly
*better* midpoint series than the live engine's, which only approximates
high/low from the tick range seen within a bucket), so a model trained
here sees an identical feature definition to what it will be fed in
production.

Label: by default, naive next-bar direction — did the close go up or down
one bar after the features were computed. This was deliberately the
simplest possible thing to try first, not a claim that it's the right
target — three rounds of adding indicators (see docs/model-training.md)
and one round of pooling 13 symbols both failed to move accuracy past a
coin flip on this label, which points at the label itself as the limiting
factor. Two knobs loosen it, and can be combined:

  --horizon N       Label bar i by the direction of the move from bar i to
                     bar i+N (default N=1, the original next-bar label).
                     A longer horizon gives price more room to make a real
                     move before averaging is forced to call a direction,
                     at the cost of overlapping (correlated) label windows
                     and fewer non-overlapping-in-spirit examples.

  --min-move X      Drop any row whose |move| over the horizon is smaller
                     than X (a fraction, e.g. 0.02 = 2%) — filters out the
                     noisy near-zero moves a 1-bar label is forced to call
                     one way or the other. Defaults to None, meaning "derive
                     it from real trading costs" — see --taker-fee /
                     --profit-margin below — rather than an arbitrary
                     number; pass this explicitly to override that.

  --top-fraction F  Instead of (or combined with) --min-move, keep only
                     the most extreme F fraction of moves (e.g. 0.3 = keep
                     the top/bottom 30%), computed *per symbol* from that
                     symbol's own horizon-move distribution — this adapts
                     to each symbol's own volatility instead of one flat
                     percentage meaning something very different for BTC
                     than for a high-volatility altcoin. When both
                     --min-move (or its derived default) and --top-fraction
                     apply, whichever threshold is larger for that symbol
                     wins, so both constraints hold.

  --taker-fee X     Kraken's spot taker fee as a fraction (default 0.008 =
                     0.80%, the entry 30-day-volume tier — see Kraken's fee
                     schedule for higher tiers). Doubled for a round trip
                     (enter + exit), and combined with --slippage, to
                     derive --min-move's default when --min-move isn't
                     given explicitly: a label that doesn't clear real
                     trading costs isn't a trade worth labeling "correct"
                     even if the price direction call was right.

  --slippage X      Expected slippage per leg as a fraction (default
                     0.0005 = 0.05%) — real spot execution rarely fills at
                     the exact last-traded price, so this is folded into
                     the same round-trip cost --taker-fee is, doubled the
                     same way, before deriving --min-move's default. A
                     label or a simulated trade that only subtracted fees
                     would still be optimistic about what a live order
                     actually nets.

  --profit-margin X Required edge *above* breakeven (a fraction, default
                     0.0), added to the round-trip cost when deriving
                     --min-move's default. E.g. --profit-margin 0.01 with
                     the default 0.008 taker fee and 0.0005 slippage
                     requires a move worth at least 1.7% (round-trip fee +
                     slippage) + 1% (margin) = 2.7% before a row counts as
                     a labeled trade.

  --label-scheme S  "fixed-horizon" (default, described above): labels by
                     the sign of the *endpoint* move at bar i+horizon only.
                     "triple-barrier": walks forward from bar i using each
                     subsequent bar's real high/low (not just its close),
                     and labels 1/0 by whichever of an upper (profit) or
                     lower (stop) barrier — both sized from the same
                     fee-derived --min-move threshold — is touched *first*
                     within --horizon bars. A bar that never touches either
                     barrier in time, or touches both within the same bar
                     (real OHLC can't say which came first intrabar), is
                     dropped rather than guessed. This is a materially
                     different (and more realistic) label than
                     fixed-horizon: it can drop a bar whose endpoint close
                     *would* have cleared --min-move, if price touched the
                     stop-loss barrier somewhere along the way and never
                     recovered — exactly what a live stop-loss order would
                     have closed out for a loss, which fixed-horizon has no
                     way to see since it only checks where price ended up.
                     See triple_barrier_label / triple_barrier_net_pnl.

  --holdout-days N  Institutional audit Phase 2.5 (sequential data-
                     snooping correction — see
                     claude/institutional-audit-2026-09-27.md): seals off
                     the most recent N days of EVERY symbol's history
                     before it ever reaches build_dataset, walk-forward
                     CV, or the production train/test split — the data
                     simply isn't there as far as this script's normal
                     sweep/train path is concerned. Default 0 disables
                     sealing (matches every round run before this option
                     existed).

                     The discipline this requires: once you start a
                     research program with --holdout-days N, use the
                     SAME N on every sweep round from then on, and never
                     look at that sealed window's data by any means
                     (not even "just to peek") until you have a specific
                     candidate configuration you're ready to call final.
                     Only then run scripts/evaluate_holdout.py — a
                     separate script, deliberately, so "sweep training"
                     and "final holdout evaluation" can never be
                     accidentally run through the same code path — against
                     that exact configuration. This is what makes the
                     holdout period a true out-of-sample test: no prior
                     round's feature/label/horizon choice, however
                     indirectly, was tuned against it.

  --embargo N       Purges/embargoes N bars at each train/test boundary —
                     applies to both --folds 1 (the single split that's
                     actually trained and saved) and --folds > 1
                     (walk-forward validation). Default None = use
                     --horizon (the label's maximum forward reach, and the
                     minimum embargo that rules out literal label overlap
                     between train and test — see walk_forward_splits'
                     docstring for the purge-then-embargo mechanics). A
                     label at bar i isn't just a function of bar i: a
                     fixed-horizon label looks all the way to bar
                     i+horizon, and triple-barrier can touch a barrier
                     anywhere in i+1..i+horizon, so without this, a
                     training row near the boundary can have a label
                     computed from data that falls inside the test set —
                     training on information that peeks into its own
                     evaluation.

This turns the label into something closer to "simulated paper trading":
instead of asking "did price go up or down" (or even "did it move a lot"),
every row that survives filtering represents a hypothetical round-trip
trade that would have cleared real transaction costs (fees + slippage).
The script goes a step further and simulates the actual money each
evaluated strategy would have made on the held-out test period — see
"Evaluated against" below — using historical data now rather than waiting
weeks/months for a live paper-trading feed to accumulate the same
information forward. With --label-scheme triple-barrier, that simulation
also exits each trade at whichever barrier is touched first (or, on
timeout, at the vertical barrier's close) instead of always holding to a
fixed bar count, which is what a live stop-loss/take-profit order would
actually do.

Evaluated against two baselines so a small accuracy edge doesn't get
oversold, on both classification accuracy AND simulated net P&L (mean
per-trade return after the same round-trip cost, on the test period):
  - majority-class baseline: always predict whichever direction was more
    common in the (post-filtering) training set
  - persistence baseline: predict the same direction as the most recent
    completed horizon-length move (a classic "trend continues" naive
    forecaster), evaluated on the same filtered rows the model is scored
    on, so it's an apples-to-apples comparison
Net P&L is the metric that actually matters for a trading strategy — a
model can have higher accuracy than a baseline and still make less money
per trade (or lose money) if its correct calls are on smaller moves and
its wrong calls are on larger ones. If the trained model can't beat both
baselines' net P&L by a meaningful margin, that's a real result to know,
not a reason to hide the run.

Train/test split is time-ordered (earliest N% train, latest test) —
never shuffled — because shuffling would let the model train on rows that
are chronologically *after* some of its test rows, which is lookahead
leakage a live model could never actually have.

Usage:
    export SUPABASE_DB_URL=postgresql://...
    python3 scripts/train_model.py --symbol BTC-USD
    python3 scripts/train_model.py --symbol BTC-USD --interval 60 --kind gboost \
        --model-out models/btc_usd_gboost.joblib

    # Pool multiple symbols' history into one combined training set,
    # instead of one model per symbol from ~700 rows each:
    python3 scripts/train_model.py --symbol BTC-USD,ETH-USD,SOL-USD
    python3 scripts/train_model.py --symbol all   # every symbol with data at --interval

    # A 12-hour-ahead label, only keeping moves >= 4.5% (or, per symbol,
    # the top/bottom 30% of that symbol's 12h moves, whichever is bigger):
    python3 scripts/train_model.py --symbol all --horizon 12 --min-move 0.045 --top-fraction 0.3

    # Walk-forward validation instead of one static 80/20 split: 5 folds,
    # each training on everything before it and testing on a later,
    # different block. No model is saved in this mode — it's a check on
    # whether a configuration holds up across multiple time periods before
    # committing to it with a normal (--folds 1, the default) run.
    python3 scripts/train_model.py --symbol all --horizon 12 --folds 5

    # Triple-barrier labels instead of a fixed-horizon endpoint check —
    # labels/evaluates against upper (profit) and lower (stop) barriers
    # touched along the path, not just where price ends up 12 bars later:
    python3 scripts/train_model.py --symbol all --horizon 12 --label-scheme triple-barrier

Pooling ("--symbol" given a comma-separated list, or the literal "all")
computes each symbol's features independently over its own close/high/low
history — every bar-derived feature here is already a ratio/normalized
value (a fraction of price, or rescaled to roughly [-1, 1]), specifically
so it's comparable across symbols at very different price levels, which
is what makes pooling reasonable rather than mixing incomparable raw
numbers. Each symbol is still time-ordered split *individually* (so a bar
from symbol A is never trained on using a chronologically later bar from
symbol A, and one symbol's split boundary never leaks into another's) and
only then are all symbols' train rows concatenated into one training set,
and all their test rows into one test set. Baselines are computed the
same pooled way: majority-class over the pooled training labels, and
persistence accuracy aggregated across every symbol's own test-period bars
(total correct / total bars, not an average of per-symbol accuracies,
so a symbol with more test bars contributes proportionally more).

Requires: psycopg2-binary (requirements-training.txt) and scikit-learn +
joblib (requirements-ml.txt).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running this script directly (python3 scripts/train_model.py) as
# well as as a module — put the package root (python-strategy/) on the
# path so `import strategy...` resolves either way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy.indicators import (
    awesome_oscillator,
    bar_momentum,
    bollinger_bandwidth,
    bollinger_percent_b,
    cci,
    ema_ratio,
    macd_histogram,
    parkinson_vol,
    realized_vol,
    returns_zscore,
    rsi,
    sma_ratio,
    volume_ratio,
    williams_percent_r,
)

FEATURE_ORDER = [
    "sma_ratio",
    "ema_ratio",
    "rsi",
    "realized_vol",
    "bar_momentum",
    "bollinger_percent_b",
    "bollinger_bandwidth",
    "awesome_oscillator",
    "macd_histogram",
    "cci",
    "williams_percent_r",
    # Volume-derived features (see strategy/indicators.py) — appended
    # rather than interleaved so an existing feature_order in
    # strategy_config.toml pointing at the first 11 stays meaningful (a
    # model trained before these existed just never saw them, not a
    # silent reordering of what it did see).
    "volume_ratio",
    "parkinson_vol",
    # Rolling Z-score of log returns — see strategy/indicators.py's
    # returns_zscore docstring for how this differs from
    # bollinger_percent_b (a price Z-score, not a returns Z-score) and
    # bar_momentum (raw, non-standardized cumulative return). Appended
    # last for the same feature_order-stability reason as the volume
    # features above.
    "returns_zscore",
]


class MissingCredentials(RuntimeError):
    pass


def connect():
    """Same SUPABASE_DB_URL-only pattern as historical-data/db.py — this
    script is intentionally standalone rather than importing across the
    historical-data/python-strategy package boundary (they're kept as
    separate dependency footprints on purpose; see docs/historical-data-pipeline.md)."""
    import psycopg2

    dsn = os.environ.get("SUPABASE_DB_URL")
    if not dsn:
        raise MissingCredentials(
            "SUPABASE_DB_URL is not set. Get the connection string from the Supabase "
            "dashboard for the Rootstock-vercel project (Project Settings -> Database "
            "-> Connection string, Session pooler) and export it — never put it in a config file."
        )
    return psycopg2.connect(dsn)


def load_ohlc(
    conn, symbol: str, interval_minutes: int
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """Returns (closes, midpoints, highs, lows, volumes) — midpoints =
    (high + low) / 2 per candle, straight from Kraken's own recorded
    high/low (real traded range), which is what awesome_oscillator is fed;
    highs/lows are the same real recorded values, needed unaveraged for
    cci (typical price = (high+low+close)/3) and williams_percent_r
    (highest-high/lowest-low over a window). This is actually a truer
    high/low series than the live engine gets by default (strategy/bars.py
    can only approximate high/low from mid-price ticks seen within a
    bucket unless a live trade feed is wired up via on_trade() — see
    bars.py's module docstring), a known, documented asymmetry, not a
    mismatch that breaks parity on the close-based features. volumes is
    the real per-candle traded volume Kraken's Trades endpoint reports
    (same quantity strategy/bars.py's on_trade() accumulates live), fed to
    volume_ratio — 0.0 for a bar with no recorded volume."""
    with conn.cursor() as cur:
        cur.execute(
            """
            select close, high, low, volume
            from ohlc_candles
            where exchange = %s and symbol = %s and interval_minutes = %s
            order by ts asc
            """,
            ("kraken", symbol, interval_minutes),
        )
        rows = cur.fetchall()
    closes = [float(r[0]) for r in rows]
    highs = [float(r[1]) for r in rows]
    lows = [float(r[2]) for r in rows]
    volumes = [float(r[3]) if r[3] is not None else 0.0 for r in rows]
    midpoints = [(h + l) / 2.0 for h, l in zip(highs, lows)]
    return closes, midpoints, highs, lows, volumes


def holdout_boundary(n_bars: int, holdout_days: int, interval_minutes: int) -> int:
    """The bar index that begins the sealed final-holdout window — bars
    `[0, boundary)` are what the normal sweep/train path is allowed to see
    when --holdout-days is set; bars `[boundary, n_bars)` are the sealed
    window scripts/evaluate_holdout.py alone is allowed to look at.

    holdout_days <= 0 (the default, matching every round run before this
    option existed) returns n_bars — i.e. nothing is sealed, the whole
    series is visible, unchanged behavior. Otherwise converts days to bars
    via interval_minutes and clamps the result to [0, n_bars] so an
    oversized --holdout-days on a short history seals everything rather
    than producing a negative slice."""
    if holdout_days <= 0:
        return n_bars
    holdout_bars = int(holdout_days * 24 * 60 / interval_minutes)
    return max(0, min(n_bars, n_bars - holdout_bars))


def seal_holdout(
    closes: list[float],
    midpoints: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float],
    holdout_days: int,
    interval_minutes: int,
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """Truncates all five parallel OHLC series to the research-visible
    prefix `[0, holdout_boundary(...))`, dropping the sealed tail entirely
    — used by load_symbol_dataset (the path every sweep/train round goes
    through) so a sealed window isn't just unused, it's structurally
    absent from anything build_dataset/walk_forward_splits/the production
    split could ever see. See holdout_boundary's docstring and the module
    docstring's --holdout-days section."""
    boundary = holdout_boundary(len(closes), holdout_days, interval_minutes)
    return closes[:boundary], midpoints[:boundary], highs[:boundary], lows[:boundary], volumes[:boundary]


def list_available_symbols(conn, interval_minutes: int) -> list[str]:
    """Every distinct symbol with at least one candle at this interval —
    used by `--symbol all` to discover the full symbol universe without
    importing historical-data/'s symbols.py (this script is deliberately
    standalone; see the module docstring). Alphabetical, so runs are
    reproducible."""
    with conn.cursor() as cur:
        cur.execute(
            """
            select distinct symbol
            from ohlc_candles
            where exchange = %s and interval_minutes = %s
            order by symbol asc
            """,
            ("kraken", interval_minutes),
        )
        return [r[0] for r in cur.fetchall()]


def features_at(
    closes: list[float],
    midpoints: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float],
    i: int,
    sma_window: int,
    ema_window: int,
    rsi_window: int,
    vol_window: int,
    bar_momentum_window: int,
    bollinger_window: int,
    bollinger_num_std: float,
    ao_fast_window: int,
    ao_slow_window: int,
    macd_fast_window: int,
    macd_slow_window: int,
    macd_signal_window: int,
    cci_window: int,
    williams_r_window: int,
) -> dict[str, float]:
    """The feature vector as of bar `i`, using the same window-slicing
    convention strategy/features.py's FeatureEngine applies live via
    BarAggregator.window()/midpoint_window()/high_window()/low_window():
    the most recent `size` completed closes (or midpoints/highs/lows) up
    to and including the current one."""
    sma_win = closes[max(0, i - sma_window + 1) : i + 1]
    ema_win = closes[max(0, i - ema_window + 1) : i + 1]
    rsi_win = closes[max(0, i - rsi_window) : i + 1]  # rsi needs window+1 closes for `window` price changes
    vol_win = closes[max(0, i - vol_window) : i + 1]
    mom_win = closes[max(0, i - bar_momentum_window + 1) : i + 1]
    boll_win = closes[max(0, i - bollinger_window + 1) : i + 1]
    ao_win = midpoints[max(0, i - ao_slow_window + 1) : i + 1]
    macd_win = closes[max(0, i - (macd_slow_window + macd_signal_window) + 1) : i + 1]
    typical_win_start = max(0, i - cci_window + 1)
    typical_prices = [
        (h + l + c) / 3.0
        for h, l, c in zip(highs[typical_win_start : i + 1], lows[typical_win_start : i + 1], closes[typical_win_start : i + 1])
    ]
    wr_start = max(0, i - williams_r_window + 1)
    wr_closes = closes[wr_start : i + 1]
    wr_highs = highs[wr_start : i + 1]
    wr_lows = lows[wr_start : i + 1]
    # Same window as realized_vol (vol_win/vol_window) — see
    # strategy/features.py's FeatureEngine, which reuses its own
    # _vol_window for these two rather than adding a separate parameter.
    volume_win = volumes[max(0, i - vol_window + 1) : i + 1]
    parkinson_highs = highs[max(0, i - vol_window + 1) : i + 1]
    parkinson_lows = lows[max(0, i - vol_window + 1) : i + 1]
    return {
        "sma_ratio": sma_ratio(sma_win),
        "ema_ratio": ema_ratio(ema_win),
        "rsi": rsi(rsi_win),
        "realized_vol": realized_vol(vol_win),
        "bar_momentum": bar_momentum(mom_win),
        "bollinger_percent_b": bollinger_percent_b(boll_win, bollinger_num_std),
        "bollinger_bandwidth": bollinger_bandwidth(boll_win, bollinger_num_std),
        "awesome_oscillator": awesome_oscillator(ao_win, ao_fast_window, ao_slow_window),
        "macd_histogram": macd_histogram(macd_win, macd_fast_window, macd_slow_window, macd_signal_window),
        "cci": cci(typical_prices),
        "williams_percent_r": williams_percent_r(wr_closes, wr_highs, wr_lows),
        "volume_ratio": volume_ratio(volume_win),
        "parkinson_vol": parkinson_vol(parkinson_highs, parkinson_lows),
        # Same window as realized_vol (vol_win) — see returns_zscore's
        # docstring for why this needs the identical log-return series.
        "returns_zscore": returns_zscore(vol_win),
    }


def dataset_warmup(
    sma_window: int,
    ema_window: int,
    rsi_window: int,
    vol_window: int,
    bar_momentum_window: int,
    bollinger_window: int,
    ao_slow_window: int,
    macd_slow_window: int,
    macd_signal_window: int,
    cci_window: int,
    williams_r_window: int,
) -> int:
    """The number of bars needed before every indicator's window has a
    full history — shared by build_dataset (to know where to start) and
    load_symbol_dataset (to know whether a symbol has enough history at
    all)."""
    return max(
        sma_window,
        ema_window,
        rsi_window + 1,
        vol_window + 1,
        bar_momentum_window,
        bollinger_window,
        ao_slow_window,
        macd_slow_window + macd_signal_window,
        cci_window,
        williams_r_window,
    )


def move(closes: list[float], i: int, horizon: int) -> float:
    """Fractional price change from bar i to bar i+horizon — the raw
    quantity both the label and the persistence baseline are built from.
    Returns 0.0 if closes[i] is 0 (degenerate, shouldn't happen with real
    price data) rather than dividing by zero."""
    if closes[i] == 0:
        return 0.0
    return (closes[i + horizon] - closes[i]) / closes[i]


def per_symbol_move_threshold(closes: list[float], warmup: int, horizon: int, top_fraction: float) -> float:
    """The |move| cutoff that keeps only the most extreme `top_fraction`
    of this symbol's horizon-length moves (e.g. top_fraction=0.3 keeps the
    most extreme 30%), computed over the same candidate bars build_dataset
    would consider. Sizing this per symbol, from that symbol's own move
    distribution, is what makes a "top fraction" threshold mean the same
    thing for a calm major (BTC) and a choppy altcoin, unlike one flat
    percentage applied to both. Returns 0.0 (no filtering) when
    top_fraction >= 1.0 or there isn't enough history to compute a
    distribution from."""
    if top_fraction >= 1.0:
        return 0.0
    magnitudes = [abs(move(closes, i, horizon)) for i in range(warmup - 1, len(closes) - horizon)]
    if not magnitudes:
        return 0.0
    magnitudes.sort()
    cutoff_index = int(len(magnitudes) * (1.0 - top_fraction))
    cutoff_index = min(max(cutoff_index, 0), len(magnitudes) - 1)
    return magnitudes[cutoff_index]


def net_pnl(closes: list[float], i: int, horizon: int, predicted_up: bool, round_trip_cost: float) -> float:
    """The simulated net return of one paper trade: go long if
    `predicted_up`, short otherwise, held from bar i to bar i+horizon,
    minus `round_trip_cost` (both legs' fees, e.g. 2x Kraken's taker fee).
    This is the actual paper-trading question — not "was the direction
    call right" but "would this specific trade have made money after real
    costs" — and can be computed directly over historical closes rather
    than waiting for a live paper-trading feed to accumulate the same
    information forward in real time."""
    actual_move = move(closes, i, horizon)
    directional_return = actual_move if predicted_up else -actual_move
    return directional_return - round_trip_cost


def simulate_net_pnl_series(closes: list[float], indices: list[int], horizon: int, predictions: list[int], round_trip_cost: float) -> list[float]:
    """The individual per-trade net P&L (see net_pnl()) for every (index,
    prediction) pair, in order — the raw series simulate_net_pnl reduces
    to a (total, count) pair, and what scripts/promotion_gate.py's
    Sharpe/deflated-Sharpe/MinTRL checks need (a return series, not just
    its sum) to evaluate a candidate model for promotion."""
    return [net_pnl(closes, i, horizon, predicted_up=bool(pred), round_trip_cost=round_trip_cost) for i, pred in zip(indices, predictions)]


def simulate_net_pnl(closes: list[float], indices: list[int], horizon: int, predictions: list[int], round_trip_cost: float) -> tuple[float, int]:
    """Total and count of simulated net P&L (see net_pnl()) across every
    (index, prediction) pair — `predictions` must be 1 (predicted up) or 0
    (predicted down), aligned 1:1 with `indices`. Returns (total, count)
    rather than a mean so callers can aggregate across symbols before
    dividing."""
    series = simulate_net_pnl_series(closes, indices, horizon, predictions, round_trip_cost)
    return sum(series), len(series)


def triple_barrier_touch(
    highs: list[float], lows: list[float], closes: list[float], i: int, max_hold: int, barrier_pct: float
) -> tuple[str | None, float | None]:
    """The core of triple-barrier labeling (--label-scheme triple-barrier):
    from a hypothetical LONG entered at closes[i], walks forward bar by bar
    (up to max_hold bars — the "vertical barrier") checking each bar's
    *real* high/low, not just its close, for whichever of two barriers is
    touched first: the upper barrier at closes[i] * (1 + barrier_pct) (a
    profit target) or the lower barrier at closes[i] * (1 - barrier_pct) (a
    stop-loss). barrier_pct is expected to already have real trading costs
    baked in (round-trip fee + slippage + any required margin — the same
    quantity --min-move derives by default for the fixed-horizon label —
    see round_trip_cost), so a touch represents an actual executable,
    cost-clearing move, not just "the price moved."

    Checking every bar's high/low along the path (rather than only the
    price at i+horizon, as the fixed-horizon label does) is what makes this
    "triple-barrier": a real stop-loss or take-profit order fires the
    moment price crosses it, not at a fixed bar count later regardless of
    what happened in between — a bar can end back above its stop-loss level
    at i+horizon and still have been legitimately stopped out along the
    way, which the fixed-horizon label has no way to see.

    Returns a (touch, long_return) pair:
      ("upper", barrier_pct)   — the profit target was touched first.
      ("lower", -barrier_pct)  — the stop-loss was touched first.
      ("timeout", final_return) — neither barrier was touched within
                                   max_hold bars; marked at bar i+max_hold's
                                   close (final_return may be any sign or
                                   magnitude smaller than barrier_pct).
      (None, None)              — not enough remaining history to look
                                   max_hold bars ahead, entry price is 0, or
                                   a bar's high AND low both cross their
                                   barrier in the same bar — real intrabar
                                   OHLC data can't say which one price
                                   actually reached first, so rather than
                                   guess, this is left unresolved for the
                                   caller to drop.

    A short position's return over the same window is exactly the negative
    of `long_return` — the exit trigger (which barrier, or the timeout) is
    identical either way, since both barriers sit at symmetric distances
    from entry."""
    if i + max_hold >= len(closes):
        return None, None
    entry = closes[i]
    if entry == 0:
        return None, None
    upper = entry * (1.0 + barrier_pct)
    lower = entry * (1.0 - barrier_pct)
    for j in range(i + 1, i + max_hold + 1):
        hit_upper = highs[j] >= upper
        hit_lower = lows[j] <= lower
        if hit_upper and hit_lower:
            return None, None  # ambiguous same-bar touch
        if hit_upper:
            return "upper", barrier_pct
        if hit_lower:
            return "lower", -barrier_pct
    return "timeout", (closes[i + max_hold] - entry) / entry


def triple_barrier_label(
    highs: list[float], lows: list[float], closes: list[float], i: int, max_hold: int, barrier_pct: float
) -> int | None:
    """1 if the upper (profit) barrier is touched before the lower (stop)
    barrier within max_hold bars, 0 if the lower is touched first, or None
    if the position times out without touching either — no cost-clearing
    move happened along the path, which build_dataset drops the row for,
    the triple-barrier analogue of --min-move dropping a too-small endpoint
    move — or the touch is ambiguous (see triple_barrier_touch)."""
    touch, _ = triple_barrier_touch(highs, lows, closes, i, max_hold, barrier_pct)
    if touch == "upper":
        return 1
    if touch == "lower":
        return 0
    return None  # "timeout" or unresolved


def triple_barrier_net_pnl(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    i: int,
    max_hold: int,
    barrier_pct: float,
    predicted_up: bool,
    round_trip_cost: float,
) -> float | None:
    """The net_pnl() analogue for --label-scheme triple-barrier: the
    simulated net return of one paper trade — long if predicted_up, short
    otherwise — exited at whichever barrier is touched first (or, on
    timeout, at the vertical barrier's close), rather than always at a
    fixed bar count later. This is a more realistic P&L simulation for a
    strategy that actually runs stop-loss/take-profit orders live. Returns
    None on the same conditions triple_barrier_touch does (insufficient
    history, zero entry price, or an ambiguous same-bar touch)."""
    touch, long_return = triple_barrier_touch(highs, lows, closes, i, max_hold, barrier_pct)
    if touch is None:
        return None
    directional_return = long_return if predicted_up else -long_return
    return directional_return - round_trip_cost


def simulate_triple_barrier_net_pnl_series(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    indices: list[int],
    horizon: int,
    barrier_pct: float,
    predictions: list[int],
    round_trip_cost: float,
) -> list[float]:
    """The individual per-trade net P&L for every (index, prediction) pair
    under --label-scheme triple-barrier — the raw series
    simulate_triple_barrier_net_pnl reduces to a (total, count) pair, and
    what scripts/promotion_gate.py needs for its Sharpe-based checks (see
    simulate_net_pnl_series's docstring for why). Rows where the touch is
    unresolved are skipped, same as simulate_triple_barrier_net_pnl."""
    out = []
    for i, pred in zip(indices, predictions):
        pnl = triple_barrier_net_pnl(highs, lows, closes, i, horizon, barrier_pct, predicted_up=bool(pred), round_trip_cost=round_trip_cost)
        if pnl is not None:
            out.append(pnl)
    return out


def simulate_triple_barrier_net_pnl(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    indices: list[int],
    horizon: int,
    barrier_pct: float,
    predictions: list[int],
    round_trip_cost: float,
) -> tuple[float, int]:
    """Same aggregation as simulate_net_pnl, but exits each simulated trade
    via triple_barrier_net_pnl (whichever barrier is touched first, or the
    vertical barrier's close) instead of always at bar i+horizon. Rows
    where the touch is unresolved are skipped rather than guessed — this
    should be rare here in practice (these are the same indices a
    triple-barrier dataset already filtered to have a resolvable touch),
    but the check is kept defensive since a caller could pass indices from
    elsewhere."""
    series = simulate_triple_barrier_net_pnl_series(highs, lows, closes, indices, horizon, barrier_pct, predictions, round_trip_cost)
    return sum(series), len(series)


def persistence_correct_and_total_from_labels(
    closes: list[float], indices: list[int], y: list[int], horizon: int
) -> tuple[int, int]:
    """Same "predict the same direction as the most recent completed
    horizon-length move" persistence baseline as
    persistence_correct_and_total, but scored against the dataset's actual
    labels `y` (aligned 1:1 with `indices`) instead of recomputing "actual
    up" from closes[i+horizon] > closes[i]. Needed for --label-scheme
    triple-barrier, where the label is which barrier was touched along the
    path, not simply the sign of the endpoint move — for the fixed-horizon
    label the two are equivalent, which is why persistence_correct_and_total
    (kept as-is, unchanged) is still what the fixed-horizon path uses."""
    correct = 0
    total = 0
    for i, actual in zip(indices, y):
        if i - horizon < 0:
            continue
        predicted_up = closes[i] > closes[i - horizon]
        correct += int(predicted_up == bool(actual))
        total += 1
    return correct, total


def build_dataset(
    closes: list[float],
    midpoints: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float],
    sma_window: int,
    ema_window: int,
    rsi_window: int,
    vol_window: int,
    bar_momentum_window: int,
    bollinger_window: int,
    bollinger_num_std: float,
    ao_fast_window: int,
    ao_slow_window: int,
    macd_fast_window: int,
    macd_slow_window: int,
    macd_signal_window: int,
    cci_window: int,
    williams_r_window: int,
    horizon: int = 1,
    min_move_threshold: float = 0.0,
    label_scheme: str = "fixed-horizon",
) -> tuple[list[list[float]], list[int], list[int]]:
    """Builds (X, y, indices) — X rows in FEATURE_ORDER, `indices` is the
    bar index `i` each row was computed as-of (needed by callers to score a
    matching persistence baseline on exactly the same rows), and y depends
    on `label_scheme`:

      "fixed-horizon" (default): y = 1 if the bar `horizon` bars after the
        features were computed closed higher, else 0. Skips any bar whose
        |move| over the horizon doesn't clear min_move_threshold — see the
        module docstring for --min-move / --top-fraction.

      "triple-barrier": y = 1 if a hypothetical long entered at bar i would
        touch an upper (profit) barrier at min_move_threshold above entry
        before a lower (stop) barrier the same distance below it, within
        `horizon` bars — 0 if the lower barrier is touched first. Checks
        every bar's real high/low along the path (see triple_barrier_label),
        not just the endpoint, so it reflects what a live stop-loss/
        take-profit order would actually experience. Skips any bar whose
        path times out without touching either barrier, or whose touch is
        ambiguous (ends up None — see triple_barrier_label).

    Either way, skips the warmup period before the largest window has a
    full history, so training isn't dominated by the neutral 0.0 values
    indicators.py returns before there's enough history."""
    warmup = dataset_warmup(
        sma_window, ema_window, rsi_window, vol_window, bar_momentum_window,
        bollinger_window, ao_slow_window, macd_slow_window, macd_signal_window,
        cci_window, williams_r_window,
    )
    X: list[list[float]] = []
    y: list[int] = []
    indices: list[int] = []
    # i is the bar the features are computed as-of; i+horizon is the label bar
    # (fixed-horizon) or the vertical barrier (triple-barrier).
    for i in range(warmup - 1, len(closes) - horizon):
        if label_scheme == "triple-barrier":
            label = triple_barrier_label(highs, lows, closes, i, horizon, min_move_threshold)
            if label is None:
                continue
        else:
            bar_move = move(closes, i, horizon)
            if abs(bar_move) < min_move_threshold:
                continue
            label = 1 if bar_move > 0 else 0
        feats = features_at(
            closes,
            midpoints,
            highs,
            lows,
            volumes,
            i,
            sma_window,
            ema_window,
            rsi_window,
            vol_window,
            bar_momentum_window,
            bollinger_window,
            bollinger_num_std,
            ao_fast_window,
            ao_slow_window,
            macd_fast_window,
            macd_slow_window,
            macd_signal_window,
            cci_window,
            williams_r_window,
        )
        X.append([feats[name] for name in FEATURE_ORDER])
        y.append(label)
        indices.append(i)
    return X, y, indices


def split_point(n: int, test_fraction: float) -> int:
    """The index that divides n time-ordered rows into (1 - test_fraction)
    train / test_fraction test, keeping at least one row on each side."""
    split = int(n * (1 - test_fraction))
    return max(1, min(n - 1, split))


def time_ordered_split(X: list, y: list, test_fraction: float):
    split = split_point(len(X), test_fraction)
    return X[:split], y[:split], X[split:], y[split:]


def _purge_train_end(indices: list[int], train_end: int, test_start_bar: int, embargo: int) -> int:
    """The largest train_end' <= train_end such that no row in
    indices[:train_end'] has index + embargo >= test_start_bar — i.e. no
    surviving training row's label lookahead (up to `embargo` bars ahead)
    reaches into the test block. Shared by walk_forward_splits (per fold
    boundary) and main()'s single time-ordered split (per symbol's own
    train/test boundary) so both apply the identical purge, rather than
    the walk-forward evaluation path being leakage-free while the actual
    production training path (--folds 1) silently isn't."""
    purged = train_end
    while purged > 0 and indices[purged - 1] + embargo >= test_start_bar:
        purged -= 1
    return purged


def _embargo_test_start(indices: list[int], test_start: int, test_end: int, test_start_bar: int, embargo: int) -> int:
    """The smallest index >= test_start such that indices[that index] >=
    test_start_bar + embargo — drops leading test rows within `embargo`
    bars of the boundary as a buffer against serial correlation across it,
    the test-side half of purge-then-embargo. See _purge_train_end for the
    train-side half."""
    start = test_start
    while start < test_end and indices[start] < test_start_bar + embargo:
        start += 1
    return start


def walk_forward_splits(X: list, y: list, indices: list, n_folds: int, embargo: int = 0):
    """Yields n_folds (X_train, y_train, idx_train, X_test, y_test, idx_test)
    tuples using an expanding-window time series split: the data is cut
    into n_folds+1 contiguous, chronologically-ordered blocks; fold k's
    (1-indexed) test set is block k+1, and its train set is every block
    before it concatenated together. So each fold's model never trains on
    data chronologically after what it's tested on (no lookahead leakage,
    same discipline as time_ordered_split), and each successive fold
    evaluates on a *different* period rather than always the same tail-end
    slice a single static split would use — the gap this closes is: even
    with more regime-diverse training data, a single 80/20 split's test set
    is always whatever the most recent ~20% happens to be (right now, one
    all-up month), so the evaluation itself is still one-directional.

    `embargo` (in bars) applies purged walk-forward cross-validation
    (Lopez de Prado) at each fold boundary, needed because a label at bar
    i isn't just a function of bar i — a fixed-horizon label looks all the
    way to bar i+horizon, and a triple-barrier label can touch a barrier
    anywhere in i+1..i+max_hold. Without an embargo, a training row near
    the boundary can have a label computed from bars that fall inside the
    test block: the model would be trained on information that peeks into
    its own evaluation period. Two purges apply, both measured in `indices`
    (real bar-index space, not row-count space, since filtering can leave
    gaps between kept rows):

    - Purge (train side): drop any trailing training row whose label
      lookahead (index + embargo) reaches at or past the first test bar's
      index, so no training label overlaps the test block at all.
    - Embargo (test side): drop any leading test row within `embargo` bars
      of the boundary, as an additional buffer against serial correlation
      across the boundary (a test bar immediately adjacent to training
      data is highly autocorrelated with it even without literal label
      overlap) — the same idea applied symmetrically, per the standard
      purge-then-embargo recipe.

    Pass `embargo=0` (the default) to disable this and get the plain
    expanding-window split — e.g. when scoring a label scheme with no
    forward lookahead of its own. In practice, callers here always pass
    the same `horizon`/`max_hold` used to build the labels, since that's
    the exact maximum forward reach a label can have.

    Yields nothing if there isn't enough data for at least one non-empty
    train and test block (n_folds+1 blocks of at least 1 row each) after
    purging/embargo is applied."""
    n = len(X)
    block_size = n // (n_folds + 1)
    if block_size < 1:
        return
    boundaries = [i * block_size for i in range(n_folds + 2)]
    boundaries[-1] = n  # last boundary absorbs any remainder from integer division
    for k in range(1, n_folds + 1):
        train_end = boundaries[k]
        test_end = boundaries[k + 1]
        if train_end >= len(indices) or test_end <= train_end:
            continue
        test_start_bar = indices[train_end]
        purged_train_end = _purge_train_end(indices, train_end, test_start_bar, embargo)
        embargoed_test_start = _embargo_test_start(indices, train_end, test_end, test_start_bar, embargo)

        X_train, y_train, idx_train = X[:purged_train_end], y[:purged_train_end], indices[:purged_train_end]
        X_test = X[embargoed_test_start:test_end]
        y_test = y[embargoed_test_start:test_end]
        idx_test = indices[embargoed_test_start:test_end]
        if not X_train or not X_test:
            continue
        yield X_train, y_train, idx_train, X_test, y_test, idx_test


def majority_class_accuracy(y_train: list[int], y_test: list[int]) -> float:
    majority = 1 if sum(y_train) >= len(y_train) / 2 else 0
    correct = sum(1 for actual in y_test if actual == majority)
    return correct / len(y_test)


def persistence_correct_and_total(closes: list[float], indices: list[int], horizon: int) -> tuple[int, int]:
    """"Predict the same direction as the most recent completed
    horizon-length move" — for label index i (predicting closes[i+horizon]
    vs closes[i]), the most recent completed move of the same length is
    closes[i] vs closes[i-horizon]. Evaluated only over the given
    `indices` (the exact rows a dataset kept, post min-move/top-fraction
    filtering) so the baseline is scored on the same bars the model is,
    not a differently-filtered set. Returns (correct, total) rather than
    a ratio so callers can aggregate across symbols before dividing."""
    correct = 0
    total = 0
    for i in indices:
        if i - horizon < 0:
            continue
        predicted_up = closes[i] > closes[i - horizon]
        actual_up = closes[i + horizon] > closes[i]
        correct += int(predicted_up == actual_up)
        total += 1
    return correct, total


def load_symbol_dataset(conn, symbol: str, interval_minutes: int, window_args: argparse.Namespace):
    """Loads and builds one symbol's (X, y, indices) dataset plus
    everything needed to time-split it and score a persistence baseline
    against it. Returns None (after printing a warning) rather than
    raising if this symbol doesn't have enough history yet — that lets
    pooling skip a thin symbol instead of aborting the whole run over one
    gap.

    `window_args` is expected to carry (in addition to the indicator
    window params) `horizon`, `min_move`, `top_fraction`, `taker_fee`,
    `profit_margin`, and `label_scheme` — see the module docstring for what
    those do. When `min_move` is None (the default), it's derived from real
    trading costs (2x taker_fee, for a round trip) plus profit_margin,
    rather than an arbitrary number. The effective per-symbol move
    threshold — used as --label-scheme fixed-horizon's endpoint-move
    threshold, or as triple-barrier's barrier width, either way the same
    fee-derived quantity — is then the *larger* of that (or the explicit
    --min-move override) and the --top-fraction cutoff computed from this
    symbol's own move distribution, so both constraints hold when both
    apply."""
    closes, midpoints, highs, lows, volumes = load_ohlc(conn, symbol, interval_minutes)

    holdout_days = getattr(window_args, "holdout_days", 0)
    if holdout_days > 0:
        total_bars = len(closes)
        closes, midpoints, highs, lows, volumes = seal_holdout(
            closes, midpoints, highs, lows, volumes, holdout_days, interval_minutes
        )
        print(
            f"[{symbol}] --holdout-days {holdout_days}: sealed off the most recent "
            f"{total_bars - len(closes)} of {total_bars} candles — this run cannot see them.",
            file=sys.stderr,
        )

    horizon = getattr(window_args, "horizon", 1)
    top_fraction = getattr(window_args, "top_fraction", 1.0)
    taker_fee = getattr(window_args, "taker_fee", 0.008)
    slippage = getattr(window_args, "slippage", 0.0005)
    profit_margin = getattr(window_args, "profit_margin", 0.0)
    label_scheme = getattr(window_args, "label_scheme", "fixed-horizon")
    # Both legs' fee + slippage — a label or a simulated trade that only
    # subtracted fees would be optimistic about what a live order actually
    # fills at (see --slippage's help text).
    round_trip_cost = 2.0 * (taker_fee + slippage)
    explicit_min_move = getattr(window_args, "min_move", None)
    min_move = explicit_min_move if explicit_min_move is not None else (round_trip_cost + profit_margin)

    warmup = dataset_warmup(
        window_args.sma_window,
        window_args.ema_window,
        window_args.rsi_window,
        window_args.vol_window,
        window_args.bar_momentum_window,
        window_args.bollinger_window,
        window_args.ao_slow_window,
        window_args.macd_slow_window,
        window_args.macd_signal_window,
        window_args.cci_window,
        window_args.williams_r_window,
    )
    min_required = warmup + horizon + 10  # a little slack beyond bare warmup so there's an actual dataset, not one row
    if len(closes) < min_required:
        print(
            f"warning: skipping {symbol} — only {len(closes)} candles at interval={interval_minutes}min, "
            f"need at least {min_required} (warmup={warmup} + horizon={horizon}, driven by whichever window is "
            "largest — ao_slow_window, macd_slow_window+macd_signal_window, etc. — + a handful of rows to train/test on).",
            file=sys.stderr,
        )
        return None

    quantile_threshold = per_symbol_move_threshold(closes, warmup, horizon, top_fraction)
    min_move_threshold = max(min_move, quantile_threshold)

    X, y, indices = build_dataset(
        closes,
        midpoints,
        highs,
        lows,
        volumes,
        window_args.sma_window,
        window_args.ema_window,
        window_args.rsi_window,
        window_args.vol_window,
        window_args.bar_momentum_window,
        window_args.bollinger_window,
        window_args.bollinger_num_std,
        window_args.ao_fast_window,
        window_args.ao_slow_window,
        window_args.macd_fast_window,
        window_args.macd_slow_window,
        window_args.macd_signal_window,
        window_args.cci_window,
        window_args.williams_r_window,
        horizon=horizon,
        min_move_threshold=min_move_threshold,
        label_scheme=label_scheme,
    )
    return {
        "symbol": symbol,
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "warmup": warmup,
        "horizon": horizon,
        "min_move_threshold": min_move_threshold,
        "barrier_pct": min_move_threshold,  # same fee-derived quantity, used as the barrier width in triple-barrier mode
        "label_scheme": label_scheme,
        "round_trip_cost": round_trip_cost,
        "X": X,
        "y": y,
        "indices": indices,
    }


def parse_intervals(raw) -> list[int]:
    """Parses --interval: a single value ("60") or a comma-separated list
    ("60,240,360,1440"), preserving order and dropping duplicates."""
    out: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v <= 0:
            raise ValueError(f"interval must be positive, got {v}")
        if v not in out:
            out.append(v)
    if not out:
        raise ValueError("--interval needs at least one value")
    return out


def resolve_symbols(conn, symbol_arg: str, interval_minutes: int) -> list[str]:
    """`--symbol` accepts a single symbol (e.g. "BTC-USD"), a
    comma-separated list ("BTC-USD,ETH-USD"), or the literal "all" — every
    symbol that has at least one candle at this interval, discovered via
    list_available_symbols() rather than importing historical-data/'s
    symbols.py (this script stays standalone; see module docstring)."""
    if symbol_arg.strip().lower() == "all":
        symbols = list_available_symbols(conn, interval_minutes)
        if not symbols:
            print(f"error: no symbols have candles at interval={interval_minutes}min.", file=sys.stderr)
            sys.exit(1)
        return symbols
    return [s.strip() for s in symbol_arg.split(",") if s.strip()]


def _train_and_evaluate(
    X_train: list,
    y_train: list,
    X_test: list,
    y_test: list,
    per_symbol_test: list[dict],
    persistence_correct: int,
    persistence_total: int,
    kind: str,
) -> dict:
    """Trains one model of the given `kind` on (X_train, y_train) and scores
    it — on both classification accuracy and simulated net P&L — against
    the held-out (X_test, y_test) plus the majority-class and persistence
    baselines. `per_symbol_test` carries, per pooled symbol, the (closes,
    horizon, round_trip_cost, idx_test, start, end) needed to slice pooled
    predictions back out per symbol for net-P&L simulation, plus — only
    when that symbol's `label_scheme` is "triple-barrier" — `highs`,
    `lows`, and `barrier_pct`, needed to simulate P&L via the same
    barrier-touch exit logic the labels themselves were built from (see
    simulate_triple_barrier_net_pnl) rather than the fixed-horizon exit
    (see simulate_net_pnl). This is the one evaluation implementation
    shared by the single-split path and every walk-forward fold, so both
    call one tested codepath rather than risk two copies quietly drifting
    apart. Returns a metrics dict; does not save the model — the caller
    decides whether and where to persist it."""
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
    else:
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier()

    model.fit(X_train, y_train)
    model_accuracy = model.score(X_test, y_test)
    baseline_accuracy = majority_class_accuracy(y_train, y_test)
    persistence_baseline = persistence_correct / persistence_total if persistence_total else 0.0

    model_predictions = list(model.predict(X_test))
    majority_class = 1 if sum(y_train) >= len(y_train) / 2 else 0

    model_pnl_total = majority_pnl_total = persistence_pnl_total = 0.0
    model_pnl_count = majority_pnl_count = persistence_pnl_count = 0
    model_pnl_series: list[float] = []
    for entry in per_symbol_test:
        triple_barrier = entry.get("label_scheme") == "triple-barrier"

        def _simulate(indices: list[int], predictions: list[int]) -> tuple[float, int]:
            if triple_barrier:
                return simulate_triple_barrier_net_pnl(
                    entry["highs"], entry["lows"], entry["closes"], indices, entry["horizon"],
                    entry["barrier_pct"], predictions, entry["round_trip_cost"],
                )
            return simulate_net_pnl(entry["closes"], indices, entry["horizon"], predictions, entry["round_trip_cost"])

        def _simulate_series(indices: list[int], predictions: list[int]) -> list[float]:
            if triple_barrier:
                return simulate_triple_barrier_net_pnl_series(
                    entry["highs"], entry["lows"], entry["closes"], indices, entry["horizon"],
                    entry["barrier_pct"], predictions, entry["round_trip_cost"],
                )
            return simulate_net_pnl_series(entry["closes"], indices, entry["horizon"], predictions, entry["round_trip_cost"])

        sym_predictions = model_predictions[entry["start"] : entry["end"]]
        total, count = _simulate(entry["idx_test"], sym_predictions)
        model_pnl_total += total
        model_pnl_count += count
        # Only the MODEL's own per-trade series is needed for
        # scripts/promotion_gate.py's checks (it evaluates the candidate
        # being considered for promotion, not the naive baselines) —
        # collected across every pooled symbol into one series, same as
        # model_pnl_total/model_pnl_count already pool across symbols.
        model_pnl_series.extend(_simulate_series(entry["idx_test"], sym_predictions))

        majority_predictions = [majority_class] * len(entry["idx_test"])
        total, count = _simulate(entry["idx_test"], majority_predictions)
        majority_pnl_total += total
        majority_pnl_count += count

        persistence_predictions = [
            1 if entry["closes"][i] > entry["closes"][i - entry["horizon"]] else 0
            for i in entry["idx_test"]
            if i - entry["horizon"] >= 0
        ]
        persistence_indices = [i for i in entry["idx_test"] if i - entry["horizon"] >= 0]
        total, count = _simulate(persistence_indices, persistence_predictions)
        persistence_pnl_total += total
        persistence_pnl_count += count

    return {
        "model": model,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "model_accuracy": model_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "persistence_baseline": persistence_baseline,
        "model_mean_pnl": model_pnl_total / model_pnl_count if model_pnl_count else 0.0,
        "majority_mean_pnl": majority_pnl_total / majority_pnl_count if majority_pnl_count else 0.0,
        "persistence_mean_pnl": persistence_pnl_total / persistence_pnl_count if persistence_pnl_count else 0.0,
        "model_pnl_count": model_pnl_count,
        "model_pnl_total": model_pnl_total,
        "model_pnl_series": model_pnl_series,
        "majority_pnl_count": majority_pnl_count,
        "majority_pnl_total": majority_pnl_total,
        "persistence_pnl_count": persistence_pnl_count,
        "persistence_pnl_total": persistence_pnl_total,
    }


def _print_fold_report(fold_idx: int, n_folds: int, metrics: dict) -> None:
    label = f"fold {fold_idx + 1}/{n_folds}"
    print(f"[{label}] {metrics['n_train']} train / {metrics['n_test']} test rows", file=sys.stderr)
    print(f"  model accuracy:              {metrics['model_accuracy']:.3f}", file=sys.stderr)
    print(f"  majority-class baseline:     {metrics['baseline_accuracy']:.3f}", file=sys.stderr)
    print(f"  persistence baseline:        {metrics['persistence_baseline']:.3f}", file=sys.stderr)
    print(f"  model net P&L:               {metrics['model_mean_pnl']:+.4f} ({metrics['model_pnl_count']} trades)", file=sys.stderr)
    print(f"  majority-class net P&L:      {metrics['majority_mean_pnl']:+.4f} ({metrics['majority_pnl_count']} trades)", file=sys.stderr)
    print(f"  persistence net P&L:         {metrics['persistence_mean_pnl']:+.4f} ({metrics['persistence_pnl_count']} trades)", file=sys.stderr)


def _print_walk_forward_summary(fold_results: list[dict]) -> None:
    n = len(fold_results)

    def avg(key: str) -> float:
        return sum(m[key] for m in fold_results) / n

    beat_accuracy = sum(1 for m in fold_results if m["model_accuracy"] > max(m["baseline_accuracy"], m["persistence_baseline"]))
    beat_pnl = sum(1 for m in fold_results if m["model_mean_pnl"] > max(m["majority_mean_pnl"], m["persistence_mean_pnl"]))
    print(f"--- walk-forward summary across {n} fold(s) ---", file=sys.stderr)
    print(f"  avg model accuracy:          {avg('model_accuracy'):.3f}  (beat both baselines in {beat_accuracy}/{n} folds)", file=sys.stderr)
    print(f"  avg majority-class baseline: {avg('baseline_accuracy'):.3f}", file=sys.stderr)
    print(f"  avg persistence baseline:    {avg('persistence_baseline'):.3f}", file=sys.stderr)
    print(f"  avg model net P&L:           {avg('model_mean_pnl'):+.4f}  (beat both baselines' net P&L in {beat_pnl}/{n} folds)", file=sys.stderr)
    print(f"  avg majority-class net P&L:  {avg('majority_mean_pnl'):+.4f}", file=sys.stderr)
    print(f"  avg persistence net P&L:     {avg('persistence_mean_pnl'):+.4f}", file=sys.stderr)
    if beat_accuracy < n or beat_pnl < n:
        print(
            "  WARNING: model did not beat both naive baselines (accuracy and/or net P&L) in every "
            "fold — a result that only looks good on one time period isn't a validated trading model. "
            "See docs/model-training.md.",
            file=sys.stderr,
        )
    print(
        "  note: walk-forward is a validation tool — no model file is saved here. "
        "Once a configuration looks good across folds, run again with --folds 1 (the default) "
        "to train and save the production model on the full time-ordered split.",
        file=sys.stderr,
    )


def run_walk_forward(datasets: list[dict], args: argparse.Namespace) -> None:
    """Runs args.folds expanding-window folds (see walk_forward_splits) for
    each dataset independently, then pools per fold index across symbols
    exactly the way the single-split path pools across symbols for one
    split — every fold is a full miniature run: a fresh model trained and
    evaluated against both baselines, on both accuracy and net P&L, on a
    different held-out period than every other fold. A symbol with too
    little history for args.folds folds is skipped with a warning rather
    than aborting the whole run; a symbol that runs out of usable folds
    before others (rare, only with very uneven per-symbol history) simply
    doesn't contribute to those later folds. Ends with a summary averaged
    across whichever folds actually produced a usable split."""
    per_symbol_folds = []
    for d in datasets:
        # --embargo defaults to this symbol's own horizon (the label's
        # maximum forward reach — see walk_forward_splits' docstring for
        # why that's the right minimum), letting an explicit --embargo
        # only ever add *extra* margin, never less than what's needed to
        # rule out literal label overlap across the fold boundary.
        embargo = args.embargo if args.embargo is not None else d["horizon"]
        folds = list(walk_forward_splits(d["X"], d["y"], d["indices"], args.folds, embargo=embargo))
        if not folds:
            print(
                f"warning: {d['symbol']} has too little data for {args.folds} walk-forward folds — "
                "skipped from walk-forward evaluation.",
                file=sys.stderr,
            )
            continue
        per_symbol_folds.append((d, folds))

    if not per_symbol_folds:
        print(f"error: no symbol had enough history for even one walk-forward fold at --folds {args.folds}.", file=sys.stderr)
        sys.exit(1)

    fold_results = []
    for fold_idx in range(args.folds):
        X_train, y_train, X_test, y_test = [], [], [], []
        persistence_correct = persistence_total = 0
        per_symbol_test: list[dict] = []
        for d, folds in per_symbol_folds:
            if fold_idx >= len(folds):
                continue  # this symbol ran out of usable folds before the others did
            sym_X_train, sym_y_train, _sym_idx_train, sym_X_test, sym_y_test, sym_idx_test = folds[fold_idx]
            X_train.extend(sym_X_train)
            y_train.extend(sym_y_train)
            test_start = len(X_test)
            X_test.extend(sym_X_test)
            y_test.extend(sym_y_test)
            per_symbol_test.append({
                "symbol": d["symbol"],
                "closes": d["closes"],
                "highs": d["highs"],
                "lows": d["lows"],
                "horizon": d["horizon"],
                "round_trip_cost": d["round_trip_cost"],
                "barrier_pct": d["barrier_pct"],
                "label_scheme": d["label_scheme"],
                "idx_test": sym_idx_test,
                "start": test_start,
                "end": len(X_test),
            })
            if d["label_scheme"] == "triple-barrier":
                correct, total = persistence_correct_and_total_from_labels(d["closes"], sym_idx_test, sym_y_test, d["horizon"])
            else:
                correct, total = persistence_correct_and_total(d["closes"], sym_idx_test, d["horizon"])
            persistence_correct += correct
            persistence_total += total

        if len(set(y_train)) < 2 or not X_test:
            print(
                f"[fold {fold_idx + 1}/{args.folds}] skipped — training labels are all one class, "
                "or no symbol had test rows in this fold.",
                file=sys.stderr,
            )
            continue

        metrics = _train_and_evaluate(
            X_train, y_train, X_test, y_test, per_symbol_test, persistence_correct, persistence_total, args.kind
        )
        fold_results.append(metrics)
        _print_fold_report(fold_idx, args.folds, metrics)

    if not fold_results:
        print("error: no fold produced a usable train/test split.", file=sys.stderr)
        sys.exit(1)

    _print_walk_forward_summary(fold_results)


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Every CLI argument load_symbol_dataset() needs (indicator windows,
    label/horizon/cost params, --holdout-days) — shared between this
    script's main() and scripts/evaluate_holdout.py so the two scripts
    can never quietly drift apart on what a given configuration means.
    Deliberately excludes train-specific args (--kind, --model-out,
    --folds, --test-fraction, --embargo) that only main() uses."""
    parser.add_argument(
        "--symbol",
        required=True,
        help='Normalized symbol (e.g. BTC-USD), a comma-separated list to pool ("BTC-USD,ETH-USD"), '
        'or "all" to pool every symbol with data at --interval.',
    )
    parser.add_argument("--interval", type=str, default="60", help="Candle resolution in minutes (default 60, matching strategy_config.example.toml's bar_interval_minutes). Accepts a comma-separated list (e.g. 60,240,360,1440) to pool several timeframes of the same symbols together — each (symbol, interval) becomes its own dataset, walk-forward validation only (--folds > 1). Needs the coarser intervals to exist in ohlc_candles (historical-data/resample_ohlc.py).")
    parser.add_argument("--sma-window", type=int, default=20)
    parser.add_argument("--ema-window", type=int, default=12)
    parser.add_argument("--rsi-window", type=int, default=14)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--bar-momentum-window", type=int, default=10)
    parser.add_argument("--bollinger-window", type=int, default=20)
    parser.add_argument("--bollinger-num-std", type=float, default=2.0)
    parser.add_argument("--ao-fast-window", type=int, default=5)
    parser.add_argument("--ao-slow-window", type=int, default=34)
    parser.add_argument("--macd-fast-window", type=int, default=12)
    parser.add_argument("--macd-slow-window", type=int, default=26)
    parser.add_argument("--macd-signal-window", type=int, default=9)
    parser.add_argument("--cci-window", type=int, default=20)
    parser.add_argument("--williams-r-window", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=1, help="Label bar i by the direction of the move to bar i+horizon (default 1 = next-bar direction).")
    parser.add_argument("--min-move", type=float, default=None, help="Drop rows whose |move| over --horizon is smaller than this fraction (e.g. 0.02 = 2%%) (--label-scheme fixed-horizon), or the triple-barrier width (--label-scheme triple-barrier). Default None = derive it from real trading costs (2x --taker-fee + --slippage + --profit-margin) instead of an arbitrary number.")
    parser.add_argument("--top-fraction", type=float, default=1.0, help="Keep only the most extreme fraction of each symbol's moves (e.g. 0.3 = top/bottom 30%%), computed per symbol. Default 1.0 = no filtering. Combined with --min-move (or its derived default) via max() when both are set.")
    parser.add_argument("--taker-fee", type=float, default=0.008, help="Kraken's spot taker fee as a fraction (default 0.008 = 0.80%%, the entry 30-day-volume tier). Doubled for a round trip, used to derive --min-move's default.")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Expected slippage per leg as a fraction (default 0.0005 = 0.05%%) — an allowance for the fill price differing from the quoted price, doubled for a round trip just like --taker-fee, and added into --min-move's derived default alongside it. Real spot execution rarely fills at the exact last-traded price, so a label that only subtracts fees is still optimistic about what a live order would net.")
    parser.add_argument("--profit-margin", type=float, default=0.0, help="Required edge above breakeven (a fraction, default 0.0), added to round-trip cost when deriving --min-move's default.")
    parser.add_argument(
        "--label-scheme",
        choices=["fixed-horizon", "triple-barrier"],
        default="fixed-horizon",
        help='"fixed-horizon" (default): label by the sign of the move to bar i+horizon, filtered by '
        "--min-move/--top-fraction on that endpoint move only (see the module docstring). "
        '"triple-barrier": walk forward from bar i using each subsequent bar\'s real high/low (not '
        "just its close), and label 1/0 by whichever of an upper (profit) or lower (stop) barrier is "
        "touched *first* within --horizon bars — both barriers sized from the same fee-derived "
        "--min-move threshold fixed-horizon uses as its endpoint-move cutoff. A bar whose path never "
        "touches either barrier in time (a timeout), or that touches both within the same bar (real "
        "OHLC data can't say which came first intrabar), is dropped rather than guessed. This tracks "
        "what a live strategy running real stop-loss/take-profit orders would actually experience, "
        "instead of only checking where price ended up at a fixed bar count later — see "
        "triple_barrier_label/triple_barrier_net_pnl.",
    )
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=0,
        help="Institutional audit Phase 2.5: seal off the most recent N days of every symbol's "
        "history from this run entirely (0 = disabled, the default). See the module docstring's "
        "--holdout-days section for the discipline this requires and "
        "scripts/evaluate_holdout.py for the separate, one-time evaluation step against the "
        "sealed window.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_args(parser)
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of each symbol's (time-ordered) data held out for testing.")
    parser.add_argument(
        "--folds",
        type=int,
        default=1,
        help="Default 1 = the original single time-ordered 80/20 split, trained and saved as usual. "
        "N > 1 runs an expanding-window walk-forward validation instead (see walk_forward_splits): "
        "N folds, each training on all data before it and testing on a different, later block, so "
        "the evaluation isn't always scored on just the most recent slice. Validation only — no "
        "model is saved in this mode.",
    )
    parser.add_argument(
        "--embargo",
        type=int,
        default=None,
        help="Bars purged/embargoed at each train/test boundary to prevent label leakage across it — "
        "applies to both --folds 1 (the split that's actually trained and saved) and --folds > 1 "
        "(walk-forward validation); see walk_forward_splits' docstring for the purge-then-embargo "
        "mechanics. Default None = use each symbol's own --horizon (the label's maximum forward "
        "reach, and the minimum embargo that rules out literal label overlap between train and "
        "test). Pass an explicit value only to widen the buffer further (e.g. for extra margin "
        "against serial correlation across the boundary); passing something smaller than --horizon "
        "reopens the leakage this option exists to close, so there's rarely a reason to.",
    )
    parser.add_argument("--kind", choices=["logistic", "gboost"], default="logistic")
    parser.add_argument("--model-out", default=None, help="Defaults to models/<symbol>_<kind>.joblib, or models/pooled_<kind>.joblib when pooling more than one symbol.")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Run scripts/promotion_gate.py's deflated-Sharpe/MinTRL/PBO checks against this run's own simulated "
        "net-P&L series (institutional audit Phase 3.2) before saving the model, and print a PASS/FAIL verdict. "
        "Only meaningful with --folds 1 (the production save path) — a no-op warning otherwise.",
    )
    parser.add_argument(
        "--gate-enforce",
        action="store_true",
        help="With --gate: exit 1 and refuse to save the model on a FAIL verdict, instead of just warning. "
        "Off by default so --gate can be used to observe the gate's verdict without blocking a save.",
    )
    parser.add_argument(
        "--gate-trial-sharpes-file",
        default=None,
        help="JSON file: Sharpe ratios of every configuration tried in the sweep that produced this candidate "
        "(including its own), for the deflated Sharpe ratio's multiple-testing correction. Omitting this "
        "disables that correction (N=1) and is flagged in the gate's output as understating overfitting risk.",
    )
    parser.add_argument("--gate-min-dsr", type=float, default=None, help="Override promotion_gate's default minimum Deflated Sharpe Ratio (0.95).")
    parser.add_argument("--gate-max-pbo", type=float, default=None, help="Override promotion_gate's default maximum PBO (0.5).")
    args = parser.parse_args()

    try:
        import joblib
    except ImportError:
        print("error: joblib is required (pip install -r requirements-ml.txt)", file=sys.stderr)
        sys.exit(1)

    conn = connect()
    try:
        intervals = parse_intervals(args.interval)
        multi_interval = len(intervals) > 1
        if multi_interval and args.folds <= 1:
            print("error: pooling multiple --interval values is validation-only — use --folds > 1 (a saved model must target one bar interval).", file=sys.stderr)
            sys.exit(1)
        datasets = []
        for iv in intervals:
            for s in resolve_symbols(conn, args.symbol, iv):
                d = load_symbol_dataset(conn, s, iv, args)
                if d is None:
                    continue
                if multi_interval:
                    d["symbol"] = f"{s}@{iv}"
                datasets.append(d)
    finally:
        conn.close()

    if not datasets:
        print("error: no symbol had enough history to build a dataset.", file=sys.stderr)
        sys.exit(1)

    if args.folds > 1:
        if args.gate:
            print("warning: --gate has no effect with --folds > 1 (validation-only, no model saved) — re-run with --folds 1 to gate a production save.", file=sys.stderr)
        run_walk_forward(datasets, args)
        return

    pooling = len(datasets) > 1

    # Time-order split each symbol *individually* first (so a later bar
    # from one symbol never trains on an earlier held-out bar from that
    # same symbol, and one symbol's split boundary never leaks into
    # another's), then concatenate every symbol's train rows together and
    # every symbol's test rows together into one pooled dataset. Purged +
    # embargoed exactly like walk_forward_splits (see _purge_train_end/
    # _embargo_test_start) — this is the path that actually trains and
    # saves the production model, so it gets the identical leakage
    # protection walk-forward validation does, not just the evaluation
    # tool.
    X_train: list[list[float]] = []
    y_train: list[int] = []
    X_test: list[list[float]] = []
    y_test: list[int] = []
    persistence_correct = 0
    persistence_total = 0
    # Per-symbol test-set bookkeeping (closes, indices, horizon, cost, and
    # the [start, end) slice into the pooled X_test/y_test) so predictions
    # from a single pooled model.predict(X_test) call can be sliced back out
    # per symbol afterwards to simulate net P&L against that symbol's own
    # closes/horizon/cost.
    per_symbol_test: list[dict] = []
    for d in datasets:
        split = split_point(len(d["X"]), args.test_fraction)
        embargo = args.embargo if args.embargo is not None else d["horizon"]
        if 0 < split < len(d["indices"]):
            test_start_bar = d["indices"][split]
            train_end = _purge_train_end(d["indices"], split, test_start_bar, embargo)
            sym_test_start = _embargo_test_start(d["indices"], split, len(d["indices"]), test_start_bar, embargo)
        else:
            train_end = sym_test_start = split
        sym_X_train, sym_y_train, sym_idx_train = d["X"][:train_end], d["y"][:train_end], d["indices"][:train_end]
        sym_X_test, sym_y_test, sym_idx_test = (
            d["X"][sym_test_start:], d["y"][sym_test_start:], d["indices"][sym_test_start:]
        )
        X_train.extend(sym_X_train)
        y_train.extend(sym_y_train)
        test_start = len(X_test)
        X_test.extend(sym_X_test)
        y_test.extend(sym_y_test)
        per_symbol_test.append({
            "symbol": d["symbol"],
            "closes": d["closes"],
            "highs": d["highs"],
            "lows": d["lows"],
            "horizon": d["horizon"],
            "round_trip_cost": d["round_trip_cost"],
            "barrier_pct": d["barrier_pct"],
            "label_scheme": d["label_scheme"],
            "idx_test": sym_idx_test,
            "start": test_start,
            "end": len(X_test),
        })

        if d["label_scheme"] == "triple-barrier":
            correct, total = persistence_correct_and_total_from_labels(d["closes"], sym_idx_test, sym_y_test, d["horizon"])
        else:
            correct, total = persistence_correct_and_total(d["closes"], sym_idx_test, d["horizon"])
        persistence_correct += correct
        persistence_total += total

        threshold_note = f", min_move_threshold={d['min_move_threshold']:.4f}" if d["min_move_threshold"] > 0 else ""
        print(f"[{d['symbol']}] {len(d['closes'])} candles, "
              f"{len(sym_X_train)} train / {len(sym_X_test)} test rows (post-filtering{threshold_note})", file=sys.stderr)

    if len(set(y_train)) < 2:
        print("error: training labels are all one class — can't train a classifier on this window.", file=sys.stderr)
        sys.exit(1)
    if not X_test:
        print(
            "error: no test rows survived the train/test split's purge+embargo — --test-fraction is too small "
            "relative to --embargo (or its --horizon-derived default) for this much data. Increase "
            "--test-fraction or lower --embargo.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Same evaluation every walk-forward fold uses (see _train_and_evaluate) —
    # one tested implementation, not a parallel copy that could drift.
    metrics = _train_and_evaluate(
        X_train, y_train, X_test, y_test, per_symbol_test, persistence_correct, persistence_total, args.kind
    )
    model = metrics["model"]
    model_accuracy = metrics["model_accuracy"]
    baseline_accuracy = metrics["baseline_accuracy"]
    persistence_baseline = metrics["persistence_baseline"]
    model_mean_pnl = metrics["model_mean_pnl"]
    majority_mean_pnl = metrics["majority_mean_pnl"]
    persistence_mean_pnl = metrics["persistence_mean_pnl"]
    model_pnl_count, model_pnl_total = metrics["model_pnl_count"], metrics["model_pnl_total"]
    majority_pnl_count, majority_pnl_total = metrics["majority_pnl_count"], metrics["majority_pnl_total"]
    persistence_pnl_count, persistence_pnl_total = metrics["persistence_pnl_count"], metrics["persistence_pnl_total"]

    if args.gate:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import promotion_gate

        trial_sharpes = None
        if args.gate_trial_sharpes_file:
            with open(args.gate_trial_sharpes_file) as f:
                trial_sharpes = json.load(f)
        gate_kwargs = {}
        if args.gate_min_dsr is not None:
            gate_kwargs["min_dsr"] = args.gate_min_dsr
        if args.gate_max_pbo is not None:
            gate_kwargs["max_pbo"] = args.gate_max_pbo
        verdict = promotion_gate.evaluate_gate(metrics["model_pnl_series"], trial_sharpes=trial_sharpes, **gate_kwargs)
        promotion_gate._print_verdict(verdict)
        if not verdict.passed and args.gate_enforce:
            print("error: --gate-enforce is set and the promotion gate returned FAIL — refusing to save the model.", file=sys.stderr)
            sys.exit(1)

    if args.model_out:
        model_out = args.model_out
    elif pooling:
        model_out = f"models/pooled_{args.kind}.joblib"
    else:
        model_out = f"models/{datasets[0]['symbol'].lower().replace('-', '_')}_{args.kind}.joblib"
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)

    label = f"pooled across {len(datasets)} symbols" if pooling else datasets[0]["symbol"]
    print(f"[{label}] interval={args.interval}min, "
          f"{len(X_train)} train / {len(X_test)} test rows total (time-ordered split)", file=sys.stderr)
    print(f"  model accuracy:              {model_accuracy:.3f}", file=sys.stderr)
    print(f"  majority-class baseline:     {baseline_accuracy:.3f}", file=sys.stderr)
    print(f"  persistence baseline:        {persistence_baseline:.3f}", file=sys.stderr)
    print(f"  --- simulated net P&L per trade (after round-trip cost) ---", file=sys.stderr)
    print(f"  model:                       {model_mean_pnl:+.4f} ({model_pnl_count} trades, total {model_pnl_total:+.4f})", file=sys.stderr)
    print(f"  majority-class baseline:     {majority_mean_pnl:+.4f} ({majority_pnl_count} trades, total {majority_pnl_total:+.4f})", file=sys.stderr)
    print(f"  persistence baseline:        {persistence_mean_pnl:+.4f} ({persistence_pnl_count} trades, total {persistence_pnl_total:+.4f})", file=sys.stderr)
    print(f"  saved model to:              {model_out}", file=sys.stderr)
    print(f"  feature_order for strategy_config.toml: {FEATURE_ORDER}", file=sys.stderr)
    if model_accuracy <= max(baseline_accuracy, persistence_baseline):
        print(
            "  WARNING: model did not beat both naive baselines on this test split — "
            "do not treat this as a validated trading model. See docs/model-training.md.",
            file=sys.stderr,
        )
    if model_mean_pnl <= max(majority_mean_pnl, persistence_mean_pnl):
        print(
            "  WARNING: model did not beat both naive baselines' simulated net P&L — "
            "a model can have higher accuracy and still make less money per trade. "
            "See docs/model-training.md.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
