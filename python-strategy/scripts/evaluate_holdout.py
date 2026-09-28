#!/usr/bin/env python3
"""
Institutional audit Phase 2.5 (sequential data-snooping correction — see
claude/institutional-audit-2026-09-27.md): the ONE-TIME, final evaluation
of a saved model against the sealed final-holdout window that
scripts/train_model.py's --holdout-days option kept out of every sweep
round's reach.

This is a deliberately SEPARATE script from train_model.py, not a mode
flag on it, so a sweep session and a final-holdout evaluation can never be
run through the same code path by accident — the audit's exact complaint
was eleven-plus rounds of "try something, look at held-out accuracy,
iterate," and the fix has to make that structurally harder to repeat, not
just documented as a rule to remember.

What this does, in order:
  1. Loads the same symbol(s)' OHLC history train_model.py would, with the
     SAME --holdout-days and feature/label arguments the candidate model
     was trained with (you must pass these identically — see the
     "matching arguments" note below).
  2. Computes the exact same holdout_boundary() train_model.py used to
     seal the tail off, but this time keeps ONLY that sealed tail (plus
     enough preceding bars for indicator warmup — using pre-boundary
     closes as lookback CONTEXT for a feature is not label leakage; only
     the holdout window's own labels/moves matter for "has this data
     been looked at before").
  3. Scores the saved model's predictions against that holdout window on
     both classification accuracy and simulated net P&L (the same
     methodology train_model.py's own evaluation uses), plus a
     persistence baseline.
  4. Computes the Deflated Sharpe Ratio (see strategy/dsr.py) of the
     model's per-trade net-P&L series on the holdout, benchmarked against
     --num-trials — REQUIRED, and it must be supplied honestly (see its
     help text) — and reports a clear PASS/FAIL against --dsr-threshold
     (default 0.95, the conventional 95%-significance bar).

Matching arguments: this script does NOT re-derive what configuration the
model was trained with (the .joblib file is just the fitted
scikit-learn estimator, no metadata). You are responsible for passing the
EXACT same --interval, indicator window sizes, --horizon, --min-move (or
--taker-fee/--slippage/--profit-margin if it was derived), --top-fraction,
and --label-scheme that scripts/train_model.py used to produce this
model, or the feature vectors fed to model.predict() here won't match
what it was fit on. This is unavoidable without a training-metadata
sidecar file (a reasonable future improvement, not built here) — for now,
copy the exact CLI arguments you trained with, changing only --holdout-days
(must be the SAME value that was sealed during training, so this
evaluates precisely what was held back and nothing more) plus the new
--model/--num-trials/--dsr-threshold flags.

--num-trials must be supplied honestly and is the single most important
input to this script's verdict. It's not "how many times was this exact
config run" — it's the count of DISTINCT configurations (feature sets,
horizons, label schemes, thresholds) evaluated against data leading up to
this point in the research program, the same "eleven rounds" the audit's
own count identifies (see docs/model-training.md's round-by-round log,
which exists in no small part to make this number countable). Undercounting
it defeats the entire point of this script; when genuinely unsure, round
up rather than down.

Usage:
    export SUPABASE_DB_URL=postgresql://...
    # Train with a 60-day sealed holdout (every sweep round before this
    # used the identical --holdout-days 60):
    python3 scripts/train_model.py --symbol all --horizon 12 \
        --label-scheme triple-barrier --holdout-days 60 \
        --model-out models/candidate.joblib

    # Once ready to spend your ONE look at the sealed window:
    python3 scripts/evaluate_holdout.py --symbol all --horizon 12 \
        --label-scheme triple-barrier --holdout-days 60 \
        --model models/candidate.joblib --num-trials 12

Requires: the same dependencies as train_model.py (psycopg2-binary,
scikit-learn, joblib).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_model import (
    FEATURE_ORDER,
    add_dataset_args,
    build_dataset,
    connect,
    dataset_warmup,
    holdout_boundary,
    load_ohlc,
    per_symbol_move_threshold,
    persistence_correct_and_total,
    persistence_correct_and_total_from_labels,
    resolve_symbols,
    simulate_net_pnl,
    simulate_triple_barrier_net_pnl,
)
from strategy.dsr import deflated_sharpe_ratio, kurtosis, sharpe_ratio, skewness


def load_holdout_dataset(conn, symbol: str, interval_minutes: int, args: argparse.Namespace):
    """The mirror image of scripts/train_model.py's load_symbol_dataset:
    instead of sealing the tail off and keeping the research-visible
    prefix, this keeps the sealed tail (from holdout_boundary(...) minus
    enough warmup lookback, through the end of history) and discards the
    research-visible prefix beyond what's needed for that lookback.
    Returns None (after printing a warning) if --holdout-days wasn't
    actually set (there's nothing sealed to evaluate) or there isn't
    enough sealed history to build even a minimal dataset."""
    if args.holdout_days <= 0:
        print(
            f"error: --holdout-days must be > 0 to evaluate a holdout window "
            f"(got {args.holdout_days}) — this script only makes sense paired with the "
            "same --holdout-days value train_model.py sealed off during training.",
            file=sys.stderr,
        )
        return None

    closes, midpoints, highs, lows, volumes = load_ohlc(conn, symbol, interval_minutes)
    boundary = holdout_boundary(len(closes), args.holdout_days, interval_minutes)

    warmup = dataset_warmup(
        args.sma_window, args.ema_window, args.rsi_window, args.vol_window,
        args.bar_momentum_window, args.bollinger_window, args.ao_slow_window,
        args.macd_slow_window, args.macd_signal_window, args.cci_window, args.williams_r_window,
    )
    # Keep enough pre-boundary bars for indicator warmup at the very first
    # holdout bar -- this is historical CONTEXT for a feature computed as
    # of a holdout bar, not a held-out label itself, so it isn't snooping;
    # it's the same convention the very first row of any ordinary
    # train/test split already relies on.
    lookback_start = max(0, boundary - warmup)
    closes = closes[lookback_start:]
    midpoints = midpoints[lookback_start:]
    highs = highs[lookback_start:]
    lows = lows[lookback_start:]
    volumes = volumes[lookback_start:]
    # The boundary, re-expressed in the trimmed series' own index space --
    # build_dataset will compute features for every bar from `warmup - 1`
    # onward in this trimmed series; only bars at or past this local
    # boundary are actually inside the sealed holdout window.
    local_boundary = boundary - lookback_start

    horizon = args.horizon
    taker_fee = args.taker_fee
    slippage = args.slippage
    profit_margin = args.profit_margin
    label_scheme = args.label_scheme
    round_trip_cost = 2.0 * (taker_fee + slippage)
    min_move = args.min_move if args.min_move is not None else (round_trip_cost + profit_margin)

    min_required = warmup + horizon + 5
    if len(closes) < min_required:
        print(
            f"warning: skipping {symbol} — only {len(closes)} candles available in and around the "
            f"sealed holdout window (need at least {min_required}). Either --holdout-days is larger "
            "than what was actually sealed at training time, or this symbol doesn't have enough "
            "history yet.",
            file=sys.stderr,
        )
        return None

    quantile_threshold = per_symbol_move_threshold(closes, warmup, horizon, args.top_fraction)
    min_move_threshold = max(min_move, quantile_threshold)

    X, y, indices = build_dataset(
        closes, midpoints, highs, lows, volumes,
        args.sma_window, args.ema_window, args.rsi_window, args.vol_window,
        args.bar_momentum_window, args.bollinger_window, args.bollinger_num_std,
        args.ao_fast_window, args.ao_slow_window, args.macd_fast_window,
        args.macd_slow_window, args.macd_signal_window, args.cci_window, args.williams_r_window,
        horizon=horizon, min_move_threshold=min_move_threshold, label_scheme=label_scheme,
    )
    # Drop any row whose feature/label was computed as-of a bar still
    # before the sealed boundary (part of the warmup lookback, not the
    # holdout itself) -- this is the actual seal enforcement on this side:
    # only bars >= local_boundary count as "the holdout result."
    keep = [j for j, i in enumerate(indices) if i >= local_boundary]
    X = [X[j] for j in keep]
    y = [y[j] for j in keep]
    indices = [indices[j] for j in keep]

    if not X:
        print(
            f"warning: skipping {symbol} — no rows of the sealed holdout window survived "
            "warmup/filtering.",
            file=sys.stderr,
        )
        return None

    return {
        "symbol": symbol,
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "horizon": horizon,
        "min_move_threshold": min_move_threshold,
        "barrier_pct": min_move_threshold,
        "label_scheme": label_scheme,
        "round_trip_cost": round_trip_cost,
        "X": X,
        "y": y,
        "indices": indices,
    }


def per_trade_returns(entry: dict, predictions: list[int]) -> list[float]:
    """The per-trade net-P&L series (see train_model.py's net_pnl /
    triple_barrier_net_pnl) for one symbol's holdout predictions -- the
    raw sequence deflated_sharpe_ratio needs, not just its mean/count."""
    from scripts.train_model import net_pnl as _net_pnl
    from scripts.train_model import triple_barrier_net_pnl as _tb_net_pnl

    triple_barrier = entry["label_scheme"] == "triple-barrier"
    returns = []
    for i, pred in zip(entry["indices"], predictions):
        if triple_barrier:
            pnl = _tb_net_pnl(
                entry["highs"], entry["lows"], entry["closes"], i, entry["horizon"],
                entry["barrier_pct"], predicted_up=bool(pred), round_trip_cost=entry["round_trip_cost"],
            )
            if pnl is None:
                continue
        else:
            pnl = _net_pnl(entry["closes"], i, entry["horizon"], predicted_up=bool(pred), round_trip_cost=entry["round_trip_cost"])
        returns.append(pnl)
    return returns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_args(parser)
    parser.add_argument("--model", required=True, help="Path to the trained .joblib model to evaluate (from scripts/train_model.py's --model-out).")
    parser.add_argument(
        "--num-trials",
        type=int,
        required=True,
        help="REQUIRED. The count of distinct configurations already evaluated against data leading "
        "up to this point in the research program (see the module docstring's --num-trials note) -- "
        "this is the multiple-comparisons correction the Deflated Sharpe Ratio applies. There is no "
        "safe default; you must supply it honestly.",
    )
    parser.add_argument(
        "--dsr-threshold",
        type=float,
        default=0.95,
        help="Minimum Deflated Sharpe Ratio to call this a validated result (default 0.95, the "
        "conventional 95%%-significance bar). Below this, the result is reported but explicitly "
        "NOT called a win.",
    )
    args = parser.parse_args()

    try:
        import joblib
    except ImportError:
        print("error: joblib is required (pip install -r requirements-ml.txt)", file=sys.stderr)
        sys.exit(1)

    model = joblib.load(args.model)

    conn = connect()
    try:
        symbols = resolve_symbols(conn, args.symbol, args.interval)
        datasets = [d for d in (load_holdout_dataset(conn, s, args.interval, args) for s in symbols) if d is not None]
    finally:
        conn.close()

    if not datasets:
        print("error: no symbol had enough sealed-holdout history to evaluate.", file=sys.stderr)
        sys.exit(1)

    all_returns: list[float] = []
    model_pnl_total = model_pnl_count = 0
    persistence_pnl_total = persistence_pnl_count = 0
    model_correct = model_total = 0
    persistence_correct = persistence_total = 0

    for d in datasets:
        predictions = list(model.predict(d["X"]))
        model_correct += sum(1 for actual, pred in zip(d["y"], predictions) if actual == pred)
        model_total += len(d["y"])

        returns = per_trade_returns(d, predictions)
        all_returns.extend(returns)
        model_pnl_total += sum(returns)
        model_pnl_count += len(returns)

        if d["label_scheme"] == "triple-barrier":
            correct, total = persistence_correct_and_total_from_labels(d["closes"], d["indices"], d["y"], d["horizon"])
        else:
            correct, total = persistence_correct_and_total(d["closes"], d["indices"], d["horizon"])
        persistence_correct += correct
        persistence_total += total

        persistence_predictions = [
            1 if d["closes"][i] > d["closes"][i - d["horizon"]] else 0
            for i in d["indices"] if i - d["horizon"] >= 0
        ]
        persistence_indices = [i for i in d["indices"] if i - d["horizon"] >= 0]
        if d["label_scheme"] == "triple-barrier":
            total_pnl, count_pnl = simulate_triple_barrier_net_pnl(
                d["highs"], d["lows"], d["closes"], persistence_indices, d["horizon"],
                d["barrier_pct"], persistence_predictions, d["round_trip_cost"],
            )
        else:
            total_pnl, count_pnl = simulate_net_pnl(
                d["closes"], persistence_indices, d["horizon"], persistence_predictions, d["round_trip_cost"]
            )
        persistence_pnl_total += total_pnl
        persistence_pnl_count += count_pnl

        print(f"[{d['symbol']}] {len(d['X'])} holdout rows evaluated", file=sys.stderr)

    model_accuracy = model_correct / model_total if model_total else 0.0
    persistence_accuracy = persistence_correct / persistence_total if persistence_total else 0.0
    model_mean_pnl = model_pnl_total / model_pnl_count if model_pnl_count else 0.0
    persistence_mean_pnl = persistence_pnl_total / persistence_pnl_count if persistence_pnl_count else 0.0

    observed_sr = sharpe_ratio(all_returns)
    obs_skew = skewness(all_returns)
    obs_kurt = kurtosis(all_returns)
    dsr = deflated_sharpe_ratio(observed_sr, n_trials=args.num_trials, n_obs=len(all_returns), skew=obs_skew, kurt=obs_kurt)

    print("--- sealed final-holdout evaluation (one-time look) ---", file=sys.stderr)
    print(f"  holdout trades:              {model_pnl_count}", file=sys.stderr)
    print(f"  model accuracy:              {model_accuracy:.3f}", file=sys.stderr)
    print(f"  persistence baseline:        {persistence_accuracy:.3f}", file=sys.stderr)
    print(f"  model net P&L/trade:         {model_mean_pnl:+.4f} (total {model_pnl_total:+.4f})", file=sys.stderr)
    print(f"  persistence net P&L/trade:   {persistence_mean_pnl:+.4f} (total {persistence_pnl_total:+.4f})", file=sys.stderr)
    print(f"  observed Sharpe ratio:       {observed_sr:.4f}", file=sys.stderr)
    print(f"  Deflated Sharpe Ratio:       {dsr:.4f}  (n_trials={args.num_trials}, threshold={args.dsr_threshold})", file=sys.stderr)

    passed = (
        dsr >= args.dsr_threshold
        and model_mean_pnl > persistence_mean_pnl
        and model_accuracy > persistence_accuracy
    )
    if passed:
        print(
            "  RESULT: PASS — clears the Deflated Sharpe Ratio bar and beats the persistence "
            "baseline on both accuracy and net P&L, on data no prior sweep round ever looked at. "
            "This is the first result in this research program that can honestly be called validated.",
            file=sys.stderr,
        )
    else:
        print(
            "  RESULT: FAIL — do not call this a validated trading model. A DSR below the threshold "
            "means this could plausibly be the best of --num-trials noisy attempts rather than real "
            "skill, even if the raw accuracy/net-P&L numbers look appealing on their own. "
            "See docs/model-training.md.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
