#!/usr/bin/env python3
"""
Trains a baseline classifier on the OHLC-derived bar features
(strategy/indicators.py: sma_ratio, ema_ratio, rsi, realized_vol,
bar_momentum, bollinger_percent_b, bollinger_bandwidth, awesome_oscillator,
macd_histogram, cci, williams_percent_r) using real historical Kraken
candles from Supabase, and saves it via joblib for use as
strategy.model.kind = "sklearn" (see strategy_config.example.toml).

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
                     schedule for higher tiers). Used, doubled for a round
                     trip (enter + exit), to derive --min-move's default
                     when --min-move isn't given explicitly: a label that
                     doesn't clear real trading costs isn't a trade worth
                     labeling "correct" even if the price direction call
                     was right.

  --profit-margin X Required edge *above* breakeven (a fraction, default
                     0.0), added to the round-trip cost when deriving
                     --min-move's default. E.g. --profit-margin 0.01 with
                     the default 0.008 taker fee requires a move worth at
                     least 1.6% (round-trip cost) + 1% (margin) = 2.6%
                     before a row counts as a labeled trade.

This turns the label into something closer to "simulated paper trading":
instead of asking "did price go up or down" (or even "did it move a lot"),
every row that survives filtering represents a hypothetical round-trip
trade that would have cleared real transaction costs. The script goes a
step further and simulates the actual money each evaluated strategy would
have made on the held-out test period — see "Evaluated against" below —
using historical data now rather than waiting weeks/months for a live
paper-trading feed to accumulate the same information forward.

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
    realized_vol,
    rsi,
    sma_ratio,
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


def load_ohlc(conn, symbol: str, interval_minutes: int) -> tuple[list[float], list[float], list[float], list[float]]:
    """Returns (closes, midpoints, highs, lows) — midpoints = (high + low)
    / 2 per candle, straight from Kraken's own recorded high/low (real
    traded range), which is what awesome_oscillator is fed; highs/lows are
    the same real recorded values, needed unaveraged for cci (typical
    price = (high+low+close)/3) and williams_percent_r (highest-high/
    lowest-low over a window). This is actually a truer high/low series
    than the live engine gets (strategy/bars.py can only approximate
    high/low from mid-price ticks seen within a bucket, since there's no
    live trade feed wired up yet — see bars.py's module docstring), a
    known, documented asymmetry, not a mismatch that breaks parity on the
    close-based features."""
    with conn.cursor() as cur:
        cur.execute(
            """
            select close, high, low
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
    midpoints = [(h + l) / 2.0 for h, l in zip(highs, lows)]
    return closes, midpoints, highs, lows


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


def simulate_net_pnl(closes: list[float], indices: list[int], horizon: int, predictions: list[int], round_trip_cost: float) -> tuple[float, int]:
    """Total and count of simulated net P&L (see net_pnl()) across every
    (index, prediction) pair — `predictions` must be 1 (predicted up) or 0
    (predicted down), aligned 1:1 with `indices`. Returns (total, count)
    rather than a mean so callers can aggregate across symbols before
    dividing."""
    total = 0.0
    for i, pred in zip(indices, predictions):
        total += net_pnl(closes, i, horizon, predicted_up=bool(pred), round_trip_cost=round_trip_cost)
    return total, len(indices)


def build_dataset(
    closes: list[float],
    midpoints: list[float],
    highs: list[float],
    lows: list[float],
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
) -> tuple[list[list[float]], list[int], list[int]]:
    """Builds (X, y, indices) — X rows in FEATURE_ORDER, y = 1 if the bar
    `horizon` bars after the features were computed closed higher, else 0,
    and `indices` is the bar index `i` each row was computed as-of (needed
    by callers to score a matching persistence baseline on exactly the
    same rows). Skips the warmup period before the largest window has a
    full history, so training isn't dominated by the neutral 0.0 values
    indicators.py returns before there's enough history. When
    min_move_threshold > 0, also skips any bar whose |move| over the
    horizon doesn't clear it — see the module docstring for --min-move /
    --top-fraction."""
    warmup = dataset_warmup(
        sma_window, ema_window, rsi_window, vol_window, bar_momentum_window,
        bollinger_window, ao_slow_window, macd_slow_window, macd_signal_window,
        cci_window, williams_r_window,
    )
    X: list[list[float]] = []
    y: list[int] = []
    indices: list[int] = []
    # i is the bar the features are computed as-of; i+horizon is the label bar.
    for i in range(warmup - 1, len(closes) - horizon):
        bar_move = move(closes, i, horizon)
        if abs(bar_move) < min_move_threshold:
            continue
        feats = features_at(
            closes,
            midpoints,
            highs,
            lows,
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
        y.append(1 if bar_move > 0 else 0)
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


def walk_forward_splits(X: list, y: list, indices: list, n_folds: int):
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

    Yields nothing if there isn't enough data for at least one non-empty
    train and test block (n_folds+1 blocks of at least 1 row each)."""
    n = len(X)
    block_size = n // (n_folds + 1)
    if block_size < 1:
        return
    boundaries = [i * block_size for i in range(n_folds + 2)]
    boundaries[-1] = n  # last boundary absorbs any remainder from integer division
    for k in range(1, n_folds + 1):
        train_end = boundaries[k]
        test_end = boundaries[k + 1]
        X_train, y_train, idx_train = X[:train_end], y[:train_end], indices[:train_end]
        X_test, y_test, idx_test = X[train_end:test_end], y[train_end:test_end], indices[train_end:test_end]
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
    window params) `horizon`, `min_move`, `top_fraction`, `taker_fee`, and
    `profit_margin` — see the module docstring for what those do. When
    `min_move` is None (the default), it's derived from real trading costs
    (2x taker_fee, for a round trip) plus profit_margin, rather than an
    arbitrary number. The effective per-symbol move threshold is then the
    *larger* of that (or the explicit --min-move override) and the
    --top-fraction cutoff computed from this symbol's own move
    distribution, so both constraints hold when both apply."""
    closes, midpoints, highs, lows = load_ohlc(conn, symbol, interval_minutes)

    horizon = getattr(window_args, "horizon", 1)
    top_fraction = getattr(window_args, "top_fraction", 1.0)
    taker_fee = getattr(window_args, "taker_fee", 0.008)
    profit_margin = getattr(window_args, "profit_margin", 0.0)
    round_trip_cost = 2.0 * taker_fee
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
    )
    return {
        "symbol": symbol,
        "closes": closes,
        "warmup": warmup,
        "horizon": horizon,
        "min_move_threshold": min_move_threshold,
        "round_trip_cost": round_trip_cost,
        "X": X,
        "y": y,
        "indices": indices,
    }


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
    predictions back out per symbol for net-P&L simulation (see
    simulate_net_pnl). This is the one evaluation implementation shared by
    the single-split path and every walk-forward fold, so both call one
    tested codepath rather than risk two copies quietly drifting apart.
    Returns a metrics dict; does not save the model — the caller decides
    whether and where to persist it."""
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
    for entry in per_symbol_test:
        sym_predictions = model_predictions[entry["start"] : entry["end"]]
        total, count = simulate_net_pnl(
            entry["closes"], entry["idx_test"], entry["horizon"], sym_predictions, entry["round_trip_cost"]
        )
        model_pnl_total += total
        model_pnl_count += count

        majority_predictions = [majority_class] * len(entry["idx_test"])
        total, count = simulate_net_pnl(
            entry["closes"], entry["idx_test"], entry["horizon"], majority_predictions, entry["round_trip_cost"]
        )
        majority_pnl_total += total
        majority_pnl_count += count

        persistence_predictions = [
            1 if entry["closes"][i] > entry["closes"][i - entry["horizon"]] else 0
            for i in entry["idx_test"]
            if i - entry["horizon"] >= 0
        ]
        persistence_indices = [i for i in entry["idx_test"] if i - entry["horizon"] >= 0]
        total, count = simulate_net_pnl(
            entry["closes"], persistence_indices, entry["horizon"], persistence_predictions, entry["round_trip_cost"]
        )
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
        folds = list(walk_forward_splits(d["X"], d["y"], d["indices"], args.folds))
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
                "horizon": d["horizon"],
                "round_trip_cost": d["round_trip_cost"],
                "idx_test": sym_idx_test,
                "start": test_start,
                "end": len(X_test),
            })
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--symbol",
        required=True,
        help='Normalized symbol (e.g. BTC-USD), a comma-separated list to pool ("BTC-USD,ETH-USD"), '
        'or "all" to pool every symbol with data at --interval.',
    )
    parser.add_argument("--interval", type=int, default=60, help="Candle resolution in minutes (default 60, matching strategy_config.example.toml's bar_interval_minutes).")
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
    parser.add_argument("--min-move", type=float, default=None, help="Drop rows whose |move| over --horizon is smaller than this fraction (e.g. 0.02 = 2%%). Default None = derive it from real trading costs (2x --taker-fee + --profit-margin) instead of an arbitrary number.")
    parser.add_argument("--top-fraction", type=float, default=1.0, help="Keep only the most extreme fraction of each symbol's moves (e.g. 0.3 = top/bottom 30%%), computed per symbol. Default 1.0 = no filtering. Combined with --min-move (or its derived default) via max() when both are set.")
    parser.add_argument("--taker-fee", type=float, default=0.008, help="Kraken's spot taker fee as a fraction (default 0.008 = 0.80%%, the entry 30-day-volume tier). Doubled for a round trip, used to derive --min-move's default.")
    parser.add_argument("--profit-margin", type=float, default=0.0, help="Required edge above breakeven (a fraction, default 0.0), added to round-trip cost when deriving --min-move's default.")
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
    parser.add_argument("--kind", choices=["logistic", "gboost"], default="logistic")
    parser.add_argument("--model-out", default=None, help="Defaults to models/<symbol>_<kind>.joblib, or models/pooled_<kind>.joblib when pooling more than one symbol.")
    args = parser.parse_args()

    try:
        import joblib
    except ImportError:
        print("error: joblib is required (pip install -r requirements-ml.txt)", file=sys.stderr)
        sys.exit(1)

    conn = connect()
    try:
        symbols = resolve_symbols(conn, args.symbol, args.interval)
        datasets = [d for d in (load_symbol_dataset(conn, s, args.interval, args) for s in symbols) if d is not None]
    finally:
        conn.close()

    if not datasets:
        print("error: no symbol had enough history to build a dataset.", file=sys.stderr)
        sys.exit(1)

    if args.folds > 1:
        run_walk_forward(datasets, args)
        return

    pooling = len(datasets) > 1

    # Time-order split each symbol *individually* first (so a later bar
    # from one symbol never trains on an earlier held-out bar from that
    # same symbol, and one symbol's split boundary never leaks into
    # another's), then concatenate every symbol's train rows together and
    # every symbol's test rows together into one pooled dataset.
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
        sym_X_train, sym_y_train, sym_idx_train = d["X"][:split], d["y"][:split], d["indices"][:split]
        sym_X_test, sym_y_test, sym_idx_test = d["X"][split:], d["y"][split:], d["indices"][split:]
        X_train.extend(sym_X_train)
        y_train.extend(sym_y_train)
        test_start = len(X_test)
        X_test.extend(sym_X_test)
        y_test.extend(sym_y_test)
        per_symbol_test.append({
            "symbol": d["symbol"],
            "closes": d["closes"],
            "horizon": d["horizon"],
            "round_trip_cost": d["round_trip_cost"],
            "idx_test": sym_idx_test,
            "start": test_start,
            "end": len(X_test),
        })

        correct, total = persistence_correct_and_total(d["closes"], sym_idx_test, d["horizon"])
        persistence_correct += correct
        persistence_total += total

        threshold_note = f", min_move_threshold={d['min_move_threshold']:.4f}" if d["min_move_threshold"] > 0 else ""
        print(f"[{d['symbol']}] {len(d['closes'])} candles, "
              f"{len(sym_X_train)} train / {len(sym_X_test)} test rows (post-filtering{threshold_note})", file=sys.stderr)

    if len(set(y_train)) < 2:
        print("error: training labels are all one class — can't train a classifier on this window.", file=sys.stderr)
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
