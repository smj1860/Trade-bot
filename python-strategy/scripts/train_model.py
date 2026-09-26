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

Label (deliberately simple, NOT sophisticated): next-bar direction — did
the close go up or down one bar after the features were computed. This is
a naive baseline label, chosen to get a first honest read on whether these
features carry any predictive signal at all, not a claim that next-bar
direction is the right thing to trade on. Evaluated against two baselines
so a small accuracy edge doesn't get oversold:
  - majority-class baseline: always predict whichever direction was more
    common in the training set
  - persistence baseline: predict the same direction as the most recent
    completed move (a classic "trend continues" naive forecaster)
If the trained model can't beat both by a meaningful margin, that's a real
result to know, not a reason to hide the run.

Train/test split is time-ordered (earliest N% train, latest test) —
never shuffled — because shuffling would let the model train on rows that
are chronologically *after* some of its test rows, which is lookahead
leakage a live model could never actually have.

Usage:
    export SUPABASE_DB_URL=postgresql://...
    python3 scripts/train_model.py --symbol BTC-USD
    python3 scripts/train_model.py --symbol BTC-USD --interval 60 --kind gboost \
        --model-out models/btc_usd_gboost.joblib

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
) -> tuple[list[list[float]], list[int]]:
    """Builds (X, y) — X rows in FEATURE_ORDER, y = 1 if the bar right
    after the features were computed closed higher, else 0. Skips the
    warmup period before the largest window has a full history, so
    training isn't dominated by the neutral 0.0 values indicators.py
    returns before there's enough history (a real, if rare, condition
    live too, but not one worth over-representing in a training set)."""
    warmup = max(
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
    X: list[list[float]] = []
    y: list[int] = []
    # i is the bar the features are computed as-of; i+1 is the label bar.
    for i in range(warmup - 1, len(closes) - 1):
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
        y.append(1 if closes[i + 1] > closes[i] else 0)
    return X, y


def time_ordered_split(X: list, y: list, test_fraction: float):
    split = int(len(X) * (1 - test_fraction))
    split = max(1, min(len(X) - 1, split))  # keep at least one row on each side
    return X[:split], y[:split], X[split:], y[split:]


def majority_class_accuracy(y_train: list[int], y_test: list[int]) -> float:
    majority = 1 if sum(y_train) >= len(y_train) / 2 else 0
    correct = sum(1 for actual in y_test if actual == majority)
    return correct / len(y_test)


def persistence_accuracy(closes: list[float], warmup: int, split_index: int) -> float:
    """"Predict the same direction as the most recent completed move" —
    for label index i (predicting closes[i+1] vs closes[i]), the most
    recent completed move is closes[i] vs closes[i-1]."""
    correct = 0
    total = 0
    for i in range(warmup - 1 + split_index, len(closes) - 1):
        if i == 0:
            continue
        predicted_up = closes[i] > closes[i - 1]
        actual_up = closes[i + 1] > closes[i]
        correct += int(predicted_up == actual_up)
        total += 1
    return correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True, help="Normalized symbol (e.g. BTC-USD).")
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
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of the (time-ordered) data held out for testing.")
    parser.add_argument("--kind", choices=["logistic", "gboost"], default="logistic")
    parser.add_argument("--model-out", default=None, help="Defaults to models/<symbol>_<kind>.joblib")
    args = parser.parse_args()

    try:
        import joblib
    except ImportError:
        print("error: joblib is required (pip install -r requirements-ml.txt)", file=sys.stderr)
        sys.exit(1)

    conn = connect()
    try:
        closes, midpoints, highs, lows = load_ohlc(conn, args.symbol, args.interval)
    finally:
        conn.close()

    warmup = max(
        args.sma_window,
        args.ema_window,
        args.rsi_window + 1,
        args.vol_window + 1,
        args.bar_momentum_window,
        args.bollinger_window,
        args.ao_slow_window,
        args.macd_slow_window + args.macd_signal_window,
        args.cci_window,
        args.williams_r_window,
    )
    min_required = warmup + 10  # a little slack beyond bare warmup so there's an actual dataset, not one row
    if len(closes) < min_required:
        print(
            f"error: only {len(closes)} candles available for {args.symbol} at interval={args.interval}min, "
            f"need at least {min_required} (warmup={warmup}, driven by whichever window is largest — "
            "ao_slow_window, macd_slow_window+macd_signal_window, etc. — + a handful of rows to train/test on). "
            "Run historical-data/backfill_ohlc.py or import_csv.py for more history first.",
            file=sys.stderr,
        )
        sys.exit(1)

    X, y = build_dataset(
        closes,
        midpoints,
        highs,
        lows,
        args.sma_window,
        args.ema_window,
        args.rsi_window,
        args.vol_window,
        args.bar_momentum_window,
        args.bollinger_window,
        args.bollinger_num_std,
        args.ao_fast_window,
        args.ao_slow_window,
        args.macd_fast_window,
        args.macd_slow_window,
        args.macd_signal_window,
        args.cci_window,
        args.williams_r_window,
    )
    X_train, y_train, X_test, y_test = time_ordered_split(X, y, args.test_fraction)

    if len(set(y_train)) < 2:
        print("error: training labels are all one class — can't train a classifier on this window.", file=sys.stderr)
        sys.exit(1)

    if args.kind == "logistic":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
    else:
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier()

    model.fit(X_train, y_train)
    model_accuracy = model.score(X_test, y_test)
    baseline_accuracy = majority_class_accuracy(y_train, y_test)

    split_index = len(X_train)
    persistence_baseline = persistence_accuracy(closes, warmup, split_index)

    model_out = args.model_out or f"models/{args.symbol.lower().replace('-', '_')}_{args.kind}.joblib"
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)

    print(f"[{args.symbol}] interval={args.interval}min, {len(closes)} candles, "
          f"{len(X_train)} train / {len(X_test)} test rows (time-ordered split)", file=sys.stderr)
    print(f"  model accuracy:              {model_accuracy:.3f}", file=sys.stderr)
    print(f"  majority-class baseline:     {baseline_accuracy:.3f}", file=sys.stderr)
    print(f"  persistence baseline:        {persistence_baseline:.3f}", file=sys.stderr)
    print(f"  saved model to:              {model_out}", file=sys.stderr)
    print(f"  feature_order for strategy_config.toml: {FEATURE_ORDER}", file=sys.stderr)
    if model_accuracy <= max(baseline_accuracy, persistence_baseline):
        print(
            "  WARNING: model did not beat both naive baselines on this test split — "
            "do not treat this as a validated trading model. See docs/model-training.md.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
